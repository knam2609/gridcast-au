"""The warm-up smoke run of the detectors (m4_plan.md step 2): bugs only, never tuning.

§1 allows a smoke run only on the warm-up periods and synthetic data. Two passes, each
reading data only up to its own warm-up end:

1. **market:** everything ending by NEM midnight of ``market_first_day`` (2022-05-01), all
   detectors (D2/D3 have no band yet, since the QR quantiles start in Nov 2022);
2. **residual:** demand up to NEM midnight of ``residual_first_day`` (2023-06-01) with the
   development QR band, **D2/D3 only**. The market detectors and price lines are never
   computed on that data, which lies in their development period.

Each pass treats the last ``weeks`` of its warm-up as a pseudo-scored period. Alert rates,
events, the forests and a smaller injection draw there show that every piece runs on real
data. The windows are still filling (the market data start in Oct 2021, the QR band in Nov
2022), so nothing here predicts the development verdicts.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.anomalies.detectors import LINES, RESIDUAL
from gridcast.anomalies.injection import power_table, run_injections
from gridcast.anomalies.inputs import check_band_actuals, load_market_series, read_band
from gridcast.anomalies.recipe import recipe_for
from gridcast.anomalies.run import Detection, Split, detect
from gridcast.anomalies.series import MarketSeries
from gridcast.config import AppConfig
from gridcast.forecasting.origins import nem_midnight_utc

NO_INJECTIONS = pd.DataFrame()


@dataclass(slots=True)
class SmokePass:
    name: str
    split: Split
    n_intervals: int
    first_end: pd.Timestamp
    last_end: pd.Timestamp
    detection: Detection
    injections: pd.DataFrame
    seconds: dict[str, float]
    series: MarketSeries
    checked_actuals: int = 0


def _split(name: str, end_day: dt.date, weeks: int, residual_only: bool) -> Split:
    first = end_day - dt.timedelta(weeks=weeks)
    far = dt.date(9999, 12, 31)
    return Split(
        name=name,
        market_first_day=far if residual_only else first,
        residual_first_day=first,
        end_day=end_day,
    )


def market_pass(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    weeks: int,
    n_starts: int,
    inject: bool = True,
    version: int = 1,
) -> SmokePass:
    a = cfg.anomalies
    t0 = time.perf_counter()
    last = nem_midnight_utc(str(a.market_first_day))
    series = load_market_series(engine, cfg, region_id, last_end=last)
    if series.ends[-1] > last:
        raise AssertionError("the market smoke pass read data after the warm-up")
    t1 = time.perf_counter()
    split = _split("smoke-market", a.market_first_day, weeks, residual_only=False)
    det = detect(series, None, split, a, recipe=recipe_for(cfg, version))
    t2 = time.perf_counter()
    inj = run_injections(series, det.run, split, a, n_starts=n_starts) if inject else NO_INJECTIONS
    t3 = time.perf_counter()
    return SmokePass(
        "market",
        split,
        len(series),
        series.ends[0],
        series.ends[-1],
        det,
        inj,
        {"load": t1 - t0, "detect": t2 - t1, "inject": t3 - t2},
        series,
    )


def residual_pass(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    weeks: int,
    n_starts: int,
    inject: bool = True,
    version: int = 1,
) -> SmokePass:
    a = cfg.anomalies
    t0 = time.perf_counter()
    last = nem_midnight_utc(str(a.residual_first_day))
    series = load_market_series(engine, cfg, region_id, last_end=last)
    if series.ends[-1] > last:
        raise AssertionError("the residual smoke pass read data after the warm-up")
    band, actual = read_band(cfg, a.residual.dev_qr_batch, cfg.intervals.source_batch)
    band = band.issued_until(last)
    actual = actual[: len(band.issue_times)]
    checked = check_band_actuals(band, actual, series)
    t1 = time.perf_counter()
    split = _split("smoke-residual", a.residual_first_day, weeks, residual_only=True)
    det = detect(series, band, split, a, market=False, recipe=recipe_for(cfg, version))
    t2 = time.perf_counter()
    inj = (
        run_injections(series, det.run, split, a, n_starts=n_starts, labels=RESIDUAL)
        if inject
        else NO_INJECTIONS
    )
    t3 = time.perf_counter()
    return SmokePass(
        "residual",
        split,
        len(series),
        series.ends[0],
        series.ends[-1],
        det,
        inj,
        {"load": t1 - t0, "detect": t2 - t1, "inject": t3 - t2},
        series,
        checked,
    )


def _detector_line(name: str, sp: SmokePass, mask: np.ndarray, notable: float) -> str:
    det = sp.detection
    tiers = det.tiers[name][mask]
    if name in LINES:
        return f"   {name:26s} alerts {int((tiers > 0).sum()):5d}"
    p = det.run.pvalues[name][mask]
    valid = np.isfinite(p)
    if not valid.any():
        return f"   {name:26s} no p-values"
    n_win = det.run.n_window[name][mask][valid]
    pv = p[valid]
    return (
        f"   {name:26s} valid {valid.mean():6.1%}  "
        f"window n {int(n_win.min()):,}-{int(n_win.max()):,}  "
        f"p<=0.01 {np.mean(pv <= 0.01):6.2%}  p<={notable:g} {np.mean(pv <= notable):6.3%}  "
        f"notable {int((tiers >= 1).sum()):4d}  severe {int((tiers >= 2).sum()):3d}  "
        f"score max {np.nanmax(det.run.scores[name][mask]):.3g}"
    )


def summarise(sp: SmokePass, cfg: AppConfig) -> list[str]:
    """Plain-text lines for the console (and the handoff)."""
    det = sp.detection
    masks = sp.split.masks(det.run.ends)
    names = RESIDUAL if not det.run.market else (*det.run.recipe.detectors, *LINES)
    first = min(sp.split.market_first_day, sp.split.residual_first_day)
    out = [
        f"== {sp.name} pass: {sp.n_intervals:,} intervals, {sp.first_end} -> {sp.last_end}; "
        + ", ".join(f"{k} {v:.1f} s" for k, v in sp.seconds.items()),
        f"   pseudo-scored NEM days: {first} to {sp.split.end_day} (exclusive)",
    ]
    if sp.checked_actuals:
        out.append(f"   stored QR actuals equal market demand on {sp.checked_actuals:,} pairs")
    out += [_detector_line(d, sp, masks[d], cfg.anomalies.pvalues.notable) for d in names]
    if det.run.market:
        models = det.run.state.month_fits()
        out.append(
            f"   {det.run.recipe.multivariate}: {len(models)} monthly models; training rows "
            + ", ".join(f"{mm.month // 12}-{mm.month % 12 + 1:02d} {mm.n_train:,}" for mm in models)
        )
    cats: dict[str, int] = {}
    for e in det.events:
        for c in e.categories:
            cats[c] = cats.get(c, 0) + 1
    out.append(
        f"   events {len(det.events)} (lines only {sum(e.lines_only for e in det.events)}, "
        f"severe {sum(e.tier == 'severe' for e in det.events)}); by category: "
        + (", ".join(f"{k} {v}" for k, v in cats.items()) or "none")
    )
    table = power_table(sp.injections) if not sp.injections.empty else pd.DataFrame()
    if not table.empty:
        elig = sp.injections.groupby("label")["n_eligible"].first()
        out.append("   injections (power by size; n per size; eligible starts):")
        for label, grp in table.groupby("label", sort=False):
            cells = "  ".join(
                f"{s:+g}: {p:.0%}" for s, p in zip(grp["size"], grp["power"], strict=True)
            )
            n = int(grp["n"].iloc[0])
            out.append(f"     {label:26s} n {n:4d}  eligible {int(elig[label]):6,}  {cells}")
    return out


# --------------------------------------------------------------------------- evidence (step 3b)


def example_events(sp: SmokePass, k: int = 5) -> list[int]:
    """Fixed rule: the first event of each peak detector in time order, then the earliest
    remaining events, up to ``k`` (positions in the pass's event list)."""
    from gridcast.anomalies.evidence import peak_detector

    det = sp.detection
    chosen: list[int] = []
    seen: set[str] = set()
    for i, e in enumerate(det.events):
        name = peak_detector(e, det)
        if name not in seen:
            seen.add(name)
            chosen.append(i)
    chosen = chosen[:k]
    for i in range(len(det.events)):
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def evidence_lines(engine: Engine, cfg: AppConfig, region_id: str, sp: SmokePass) -> list[str]:
    """Step 3b's evidence smoke (user 2026-11-19): field completeness, banned-phrase counts,
    E1 on the demand-peak explanations, and 5 example summaries; nothing else, no writes."""
    from collections import Counter

    from gridcast.anomalies.evidence import REQUIRED_FIELDS, banned_in, missing_fields
    from gridcast.anomalies.evidence_inputs import run_evidence

    t0 = time.perf_counter()
    er = run_evidence(engine, cfg, region_id, "dev", sp.series, sp.detection)
    secs = time.perf_counter() - t0
    n = len(er.evidence)
    incomplete = [missing_fields(ev) for ev in er.evidence]
    banned = [banned_in(t, cfg.anomalies.evidence.banned_phrases) for t in er.summaries]
    reasons: Counter[str] = Counter()
    fields_by_reason: dict[str, set[str]] = {}
    for ev in er.evidence:
        for path in REQUIRED_FIELDS:
            node = ev
            for part in path.split("."):
                node = node[part]
            if "unavailable" in node:
                reasons[node["unavailable"]] += 1
                fields_by_reason.setdefault(node["unavailable"], set()).add(path)
    g = er.shap_gate
    lines = [
        f"== evidence, {sp.name} pass: {n} events in {secs:.1f} s",
        f"   A4 fields: {sum(not m for m in incomplete)} of {n} events complete "
        f"({len(REQUIRED_FIELDS)} fields each)"
        + (f"; first gaps: {next(m for m in incomplete if m)[:5]}" if any(incomplete) else ""),
        f"   banned phrases: {sum(bool(b) for b in banned)} summaries"
        + (f" ({sorted({p for b in banned for p in b})})" if any(banned) else ""),
        f"   SHAP at demand peaks: {g.n} pairs, E1 {'pass' if g.passed or g.n == 0 else 'FAIL'}"
        f" (forecast mismatches {g.forecast_mismatches}, max sum error {g.max_sum_error:.1e})",
        "   unavailable fields by reason (field-events; fields):",
    ]
    for reason, count in reasons.most_common():
        names = ", ".join(sorted(p.split(".", 1)[1] for p in fields_by_reason[reason]))
        lines.append(f"     {count:6d}  {reason} [{names}]")
    lines.append("   example summaries (first event of each peak detector, then earliest):")
    for i in example_events(sp):
        lines.append(f"     - {er.summaries[i]}")
    return lines

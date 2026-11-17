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

from gridcast.anomalies.detectors import DETECTORS, LINES, RESIDUAL
from gridcast.anomalies.injection import power_table, run_injections
from gridcast.anomalies.inputs import check_band_actuals, load_market_series, read_band
from gridcast.anomalies.run import Detection, Split, detect
from gridcast.config import AppConfig
from gridcast.forecasting.origins import nem_midnight_utc


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
    engine: Engine, cfg: AppConfig, region_id: str, weeks: int, n_starts: int
) -> SmokePass:
    a = cfg.anomalies
    t0 = time.perf_counter()
    last = nem_midnight_utc(str(a.market_first_day))
    series = load_market_series(engine, cfg, region_id, last_end=last)
    if series.ends[-1] > last:
        raise AssertionError("the market smoke pass read data after the warm-up")
    t1 = time.perf_counter()
    split = _split("smoke-market", a.market_first_day, weeks, residual_only=False)
    det = detect(series, None, split, a)
    t2 = time.perf_counter()
    inj = run_injections(series, det.run, split, a, n_starts=n_starts)
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
    )


def residual_pass(
    engine: Engine, cfg: AppConfig, region_id: str, weeks: int, n_starts: int
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
    det = detect(series, band, split, a, market=False)
    t2 = time.perf_counter()
    inj = run_injections(series, det.run, split, a, n_starts=n_starts, labels=RESIDUAL)
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
    names = RESIDUAL if not det.run.market else (*DETECTORS, *LINES)
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
        models = det.run.state.models.values()
        out.append(
            f"   isolation forest: {len(models)} monthly models; training rows "
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
    table = power_table(sp.injections)
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

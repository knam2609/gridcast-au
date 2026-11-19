"""Generate ``docs/m4_v1_diagnosis.md``: why v1's D1, D4 and D6 failed (after step 4's verdicts).

Reported only. It was written after the development verdicts, to find the mechanism behind
each failure before the user decided on a disclosed v2 (m4_plan.md §1, "Decided after the
development verdicts"). It reads the stored development run and recomputes **v1's own
state** (profiles, robust scales, month models) with the frozen code. No alternative
setting is scored here, so nothing in it tunes v2 against development data.

Before anything is written, the recomputed v1 scores of D1, D4 and D6 must equal the stored
ones bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.anomalies.detectors import (
    D6_FEATURES,
    DEMAND_LEVEL,
    ISOLATION_FOREST,
    PRICE_LEVEL,
    DetectorState,
    compute_scores,
    price_x,
)
from gridcast.anomalies.inputs import load_market_series
from gridcast.anomalies.report import ReportMismatch, StoredRun, load_run, pval
from gridcast.anomalies.series import MarketSeries
from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.evaluation.report import _md_table, money
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.provenance import git_sha
from gridcast.timeutils import INTERVAL, NEM_TZ

DOC_PATH = PROJECT_ROOT / "docs" / "m4_v1_diagnosis.md"
NOTABLE = 0.001
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
P05_LOCAL = ("2024-08-05 08:00", "2024-08-05 08:20")  # the missed class-P event, ± one interval
PROBE_VALUES = (5.0, 20.0, 80.0)
PROBE_FEATURES = 7  # the signed z-like features; the two curtailment shares lie in [0, 1]
EXTREME_Z = 4.0


@dataclass(slots=True)
class Inputs:
    run: StoredRun
    series: MarketSeries
    state: DetectorState
    k: float  # MAD consistency


def load_inputs(engine: Engine, cfg: AppConfig, region_id: str) -> Inputs:
    """The stored run and v1's recomputed state, after the bit-for-bit score check."""
    run = load_run(engine, cfg, region_id)
    last = nem_midnight_utc(str(cfg.backtest.holdout_start))
    series = load_market_series(engine, cfg, region_id, last_end=last)
    if len(series) != len(run.intervals):
        raise ReportMismatch("the market series differs in length from the stored run")
    scores, state = compute_scores(series, None, cfg.anomalies)
    for d in (DEMAND_LEVEL, PRICE_LEVEL, ISOLATION_FOREST):
        stored = run.intervals[f"{d}__score"].to_numpy()
        if not np.array_equal(stored, scores[d], equal_nan=True):
            raise ReportMismatch(f"recomputed {d} scores differ from the stored run")
    return Inputs(run, series, state, cfg.anomalies.profile.mad_consistency)


# --------------------------------------------------------------------------- helpers


def _mw(size: float) -> str:
    return f"{size:+,.0f} MW".replace("-", "−")


def _q(values: np.ndarray, qs: tuple[float, ...] = QUANTILES) -> np.ndarray:
    return np.nanquantile(values, qs)


def _col(run: StoredRun, d: str, what: str) -> np.ndarray:
    return run.intervals[f"{d}__{what}"].to_numpy()


def _nem_year(ends: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((ends - INTERVAL).tz_convert(NEM_TZ).year)


def _injections(run: StoredRun, detector: str, size: float) -> pd.DataFrame:
    inj = run.injections
    return inj[(inj.detector == detector) & (inj["size"] == size)]


def _start_rows(ends: pd.DatetimeIndex, starts: pd.Series) -> np.ndarray:
    rows = pd.Index(ends).get_indexer(pd.DatetimeIndex(pd.to_datetime(starts, utc=True)))
    if (rows < 0).any():
        raise ReportMismatch("an injection start is not on the run's grid")
    return rows


def _quartile_rows(scale: np.ndarray, detected: np.ndarray, extra: dict[str, np.ndarray]) -> list:
    """Power by quartile of the robust scale at the injection start."""
    edges = np.quantile(scale, [0, 0.25, 0.5, 0.75, 1])
    which = np.clip(np.searchsorted(edges, scale, side="right") - 1, 0, 3)
    rows = []
    for q in range(4):
        sel = which == q
        row = [f"{q + 1}: {edges[q]:.2f}–{edges[q + 1]:.2f}", f"{int(sel.sum()):,}"]
        row.append(f"{100 * detected[sel].mean():.1f}%")
        row += [f"{np.median(v[sel]):.1f}" for v in extra.values()]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- D4


def d4_section(inp: Inputs, cfg: AppConfig, tz: str) -> str:
    a, st, run = cfg.anomalies, inp.state, inp.run
    floor, scale_aud = a.price_level.floor, a.price_level.scale_aud
    scale = np.maximum(inp.k * st.price_profile.mad, floor)
    med = st.price_profile.median
    scored = _col(run, PRICE_LEVEL, "in_period") & np.isfinite(_col(run, PRICE_LEVEL, "p"))
    alert = scored & (_col(run, PRICE_LEVEL, "p") <= NOTABLE)
    at_floor = np.isclose(scale, floor, rtol=0, atol=1e-12)
    score = _col(run, PRICE_LEVEL, "score")

    def x(p: float) -> float:
        return float(price_x(np.array([p]), scale_aud)[0])

    alert_prices = _q(series_price(inp)[alert], (0.05, 0.25, 0.5, 0.75, 0.95))
    flip = x(50.0) - x(-50.0)
    spike = x(17500.0) - x(185.0)

    out = [
        "## D4 price level: the per-slot scale decides the ranking",
        "",
        f"D4 scores |z| of x = asinh(price / ${scale_aud:g}) against the same half-hour's median "
        f"and robust scale over the previous 28 days (floor {floor:g}). The transform is "
        f"linear for |price| below about ${scale_aud:g} and logarithmic above, so a routine "
        f"move from −$50 to +$50 is {flip:.2f} units of x, about as large as the "
        f"{17500 / 185:.0f}-fold spike from $185 to $17,500 ({spike:.2f}). Slots whose prices "
        "straddle $0 therefore get wide robust scales, and stable slots sit at the floor.",
        "",
        _md_table(
            ["Robust scale of x", *[f"q{int(100 * q)}" for q in QUANTILES]],
            [
                ["scored intervals", *[f"{v:.2f}" for v in _q(scale[scored])]],
                ["alerts (p ≤ 0.001)", *[f"{v:.2f}" for v in _q(scale[alert])]],
            ],
        ),
        "",
        f"- {100 * at_floor[alert].mean():.1f}% of D4's alerts sit at the floor scale, against "
        f"{100 * at_floor[scored].mean():.1f}% of scored intervals.",
        "- Price at D4's alerts, 5th / 25th / 50th / 75th / 95th percentile: "
        + " / ".join(money(v) for v in alert_prices)
        + f". The middle half of the alerts lie between {money(alert_prices[1])} and "
        f"{money(alert_prices[3])}/MWh.",
        f"- The smallest |z| that alerted was {np.nanmin(score[alert]):.1f}; the median alert "
        f"had |z| {np.nanmedian(score[alert]):.1f}.",
        "",
        "Injection power at the largest sizes, by quartile of the robust scale at the "
        "injection's start (the |z| shown is the injected price's at the start interval):",
        "",
    ]
    for size in (15000.0, -1000.0):
        df = _injections(run, PRICE_LEVEL, size)
        rows = _start_rows(inp.series.ends, df["start"])
        sc = scale[rows]
        z = np.abs(x(size) - med[rows]) / sc
        out += [
            f"*Price set to {money(size)} for 30 min ({100 * df.detected.mean():.1f}% detected):*",
            "",
            _md_table(
                ["Scale quartile at start", "Injections", "Detected", "Median injected |z|"],
                _quartile_rows(sc, df.detected.to_numpy(), {"z": z}),
            ),
            "",
        ]
    lo, hi = (pd.Timestamp(t, tz=tz).tz_convert("UTC") for t in P05_LOCAL)
    ends = inp.series.ends
    sel = [int(i) for i in np.flatnonzero((ends >= lo) & (ends <= hi))]
    p05 = [
        [
            ends[i].tz_convert(tz).strftime("%H:%M"),
            money(series_price(inp)[i]),
            money(scale_aud * float(np.sinh(med[i]))),
            f"{scale[i]:.2f}",
            f"{score[i]:.2f}",
            pval(float(_col(run, PRICE_LEVEL, "p")[i])),
        ]
        for i in sel
    ]
    out += [
        "*P05, the missed class-P event (5 Aug 2024, local interval ends):* the slot's median "
        f"price was about {money(scale_aud * float(np.sinh(med[sel[1]])))}, so the spike's |z| "
        f"stayed far below the smallest |z| that alerted anywhere in development "
        f"({np.nanmin(score[alert]):.1f}).",
        "",
        _md_table(["Interval end", "Price", "Slot median price", "Robust scale", "|z|", "p"], p05),
        "",
    ]
    return "\n".join(out)


def series_price(inp: Inputs) -> np.ndarray:
    return np.asarray(inp.series.price, dtype=np.float64)


# --------------------------------------------------------------------------- D1


def d1_section(inp: Inputs, cfg: AppConfig) -> str:
    a, st, run = cfg.anomalies, inp.state, inp.run
    scale = np.maximum(inp.k * st.demand_profile.mad, a.demand_level.floor_mw)
    p = _col(run, DEMAND_LEVEL, "p")
    scored = _col(run, DEMAND_LEVEL, "in_period") & np.isfinite(p)
    alert = scored & (p <= NOTABLE)
    score = _col(run, DEMAND_LEVEL, "score")
    year = _nem_year(inp.series.ends)
    years = [
        [
            str(y),
            f"{np.nanmin(score[alert & (year == y)]):.2f}",
            f"{np.nanmedian(scale[scored & (year == y)]):,.0f}",
        ]
        for y in np.unique(year[scored])
    ]
    out = [
        "## D1 demand level: limited by real demand variability",
        "",
        "D1 scores |z| of demand against the same half-hour's median and robust scale over "
        "the previous 28 days of the same day type. The scale is wide where weather moves "
        "demand by thousands of MW from day to day, and the trailing 26 weeks' tail is set by "
        "heatwaves and cold snaps, so a fixed 2,000 MW step is unusual only at stable slots. "
        "This is the detector working as designed: D2, which conditions on the weather "
        "forecast, catches far smaller steps.",
        "",
        _md_table(
            ["Robust scale, MW", *[f"q{int(100 * q)}" for q in QUANTILES]],
            [["scored intervals", *[f"{v:,.0f}" for v in _q(scale[scored])]]],
        ),
        "",
        _md_table(["NEM year", "Smallest alerting |z|", "Median robust scale, MW"], years),
        "",
    ]
    for size in (-2000.0, 2000.0):
        df = _injections(run, DEMAND_LEVEL, size)
        rows = _start_rows(inp.series.ends, df["start"])
        sc = scale[rows]
        out += [
            f"*{_mw(size)} for 1 h ({100 * df.detected.mean():.1f}% detected):*",
            "",
            _md_table(
                ["Scale quartile at start, MW", "Injections", "Detected", "Median size / scale"],
                _quartile_rows(sc, df.detected.to_numpy(), {"z": abs(size) / sc}),
            ),
            "",
        ]
    return "\n".join(out)


# --------------------------------------------------------------------------- D6


def d6_section(inp: Inputs, cfg: AppConfig) -> str:
    st, run = inp.state, inp.run
    p = _col(run, ISOLATION_FOREST, "p")
    scored = _col(run, ISOLATION_FOREST, "in_period") & np.isfinite(p)
    month = st.lay.nem_month
    monthly = pd.DataFrame({"m": month[scored], "a": p[scored] <= NOTABLE}).groupby("m")["a"].mean()
    feats = st.features
    complete = np.all(np.isfinite(feats), axis=1)
    ends = inp.series.ends
    scored_months = sorted(set(np.unique(month[scored]).tolist()) & set(st.models))
    reached = 0
    top_counts = np.zeros(PROBE_FEATURES + 1, dtype=np.int64)
    example_rows: list[list[str]] = []
    example = scored_months[len(scored_months) // 2]
    for m in scored_months:
        mm = st.models[m]
        rows = np.flatnonzero(complete & (ends >= mm.train_first_end) & (ends <= mm.train_last_end))
        train = feats[rows]
        s = mm.score(train)
        ptrain = mm.pvalues(s)
        top = train[ptrain <= NOTABLE]
        top_counts += np.bincount(
            (np.abs(top[:, :PROBE_FEATURES]) > EXTREME_Z).sum(axis=1), minlength=PROBE_FEATURES + 1
        )
        base = train[np.argsort(s)[len(s) // 2]]
        best = 1.0
        for f in range(PROBE_FEATURES):
            for v in PROBE_VALUES:
                probe = base.copy()
                probe[f] = v
                ps = mm.score(probe[None, :])
                pp = float(mm.pvalues(ps)[0])
                best = min(best, pp)
                if m == example and v in (PROBE_VALUES[0], PROBE_VALUES[-1]):
                    example_rows.append([D6_FEATURES[f], f"{v:g}", f"{ps[0]:.3f}", pval(pp)])
        reached += best <= NOTABLE
        if m == example:
            threshold = float(np.min(s[ptrain <= NOTABLE]))
            base_score = float(s[np.argsort(s)[len(s) // 2]])
    year, m0 = divmod(int(example), 12)
    out = [
        "## D6 Isolation Forest: saturation and month-to-month drift",
        "",
        "D6 is refitted at each NEM-month start on the previous 26 weeks, and its p-value is "
        "against that training window's own scores.",
        "",
        "**Drift (A1).** The share of a month's scored intervals at p ≤ 0.001 should be about "
        "0.1% if the month resembles its training window.",
        "",
        _md_table(
            ["Monthly share at p ≤ 0.001", *[f"q{int(100 * q)}" for q in QUANTILES]],
            [["months", *[f"{100 * v:.3f}%" for v in _q(monthly.to_numpy())]]],
        ),
        "",
        f"- {int((monthly > 0.002).sum())} of {len(monthly)} months are above 0.2%, and "
        f"{int((monthly == 0).sum())} have no alert at all.",
        "",
        "**Saturation (A3).** An isolation tree splits between a node's minimum and maximum, "
        "so a value beyond the training range is isolated no faster than the training "
        "extreme. A disturbance in one feature therefore scores no higher than an ordinary "
        "row with that feature at its training extreme. Probe: the median-score training row "
        "of each "
        f"month's model, with one of its {PROBE_FEATURES} signed z-features set to "
        + ", ".join(f"{v:g}" for v in PROBE_VALUES)
        + ".",
        "",
        f"- In {reached} of {len(scored_months)} monthly models, any probe reached p ≤ 0.001.",
        "- Among the training rows at p ≤ 0.001, pooled over the models, the number of "
        f"z-features beyond |{EXTREME_Z:g}|: "
        + ", ".join(f"{k}: {int(c):,}" for k, c in enumerate(top_counts) if c)
        + ".",
        "",
        f"*The {year}-{m0 + 1:02d} model (median row's score {base_score:.3f}; the p ≤ 0.001 "
        f"boundary is {threshold:.3f}):*",
        "",
        _md_table(["Feature set", "Value", "Score", "p"], example_rows),
        "",
    ]
    for size in (-2000.0, 2000.0):
        df = _injections(run, ISOLATION_FOREST, size)
        rows = _start_rows(ends, df["start"])
        jump = np.abs(size / st.change_scale[rows])
        out.append(
            f"- {_mw(size)} for 1 h: the scaled 5-minute demand change at the start moves "
            f"by a median of {np.nanmedian(jump):.1f}; {100 * df.detected.mean():.1f}% "
            "detected."
        )
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- document


def build_doc(engine: Engine, cfg: AppConfig, region_id: str) -> str:
    inp = load_inputs(engine, cfg, region_id)
    tz = cfg.region(region_id).local_tz
    run = inp.run
    head = [
        "# M4 v1: why D1, D4 and D6 failed (development diagnosis)",
        "",
        "Generated by `gridcast anomalies-diagnose-v1` from the stored development run "
        f"{run.run_id} (batch `{run.batch_id}`, anomaly config hash "
        f"`{cfg.anomalies.frozen_config_hash}`). **Reported only.** It was written after "
        "the development verdicts ([`reports/m4_anomalies.md`](../reports/m4_anomalies.md)) "
        "to find each failure's mechanism before the user decided on a disclosed v2 "
        '([`plans/m4_plan.md`](plans/m4_plan.md) §1, "Decided after the development '
        "verdicts\"). It recomputes v1's own profiles, robust scales and month models with the "
        "frozen code. No alternative setting is scored, so nothing here tunes v2 on "
        "development data.",
        "",
        "The recomputed v1 scores of D1, D4 and D6 equal the stored run's bit for bit. "
        "D3 is not examined here (the user kept it as in v1); its power by year is in the "
        "report.",
        "",
        f"Diagnosis code: `{git_sha() or 'unknown'}`.",
        "",
    ]
    return (
        "\n".join(
            [*head, d4_section(inp, cfg, tz), d6_section(inp, cfg), d1_section(inp, cfg)]
        ).rstrip()
        + "\n"
    )


def write_doc(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    DOC_PATH.write_text(build_doc(engine, cfg, region_id), encoding="utf-8")
    return DOC_PATH

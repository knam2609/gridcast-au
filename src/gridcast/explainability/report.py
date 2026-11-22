"""Generate ``reports/m4_explainability.md`` (+ figures) for M4 step 5, development only.

* **Global SHAP** of the stored LightGBM C fits from ``gridcast explain-dev``
  (:mod:`~gridcast.explainability.summary`), on the whole-forecast decomposition the user
  chose (m4_plan.md §1, "Clarified in step 5").
* **Question 5:** the demand side (SHAP by family for high-demand targets) and the price
  side (:mod:`~gridcast.explainability.associations`).

Every number comes from the stored sample or the database. Reported only: E1 is the one
criterion here. The report refuses its own text if it holds a banned causal phrase (§2.5).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sqlalchemy import Engine

from gridcast.anomalies.evidence import banned_in
from gridcast.config import AppConfig
from gridcast.evaluation.report import AXIS, GRID, INK, INK_2, MUTED, SURFACE, _md_table
from gridcast.explainability.associations import (
    TITLES,
    VARIABLES,
    Associations,
    compute_associations,
    load_association_data,
    period_bounds,
)
from gridcast.explainability.global_run import StoredSample, read_global
from gridcast.explainability.summary import ShapSummary, band_label, summarise
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

REPORT_NAME = "m4_explainability.md"
EDGES_FILE = "association_edges_dev.json"  # under the explain directory; step 6 reuses them
FIGURES = {
    "families": "m4_explain_families_by_band.png",
    "weather": "m4_explain_weather_dependence.png",
    "high": "m4_explain_price_above_300.png",
    "low": "m4_explain_price_negative.png",
}
# Categorical slots 1-2 of the validated palette: one per rate, the same in every figure.
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAND_FILL = "#dce8f7"
SHORT_TITLES = {
    "demand": "Demand, MW",
    "temperature": "ERA5 temperature, °C",
    "wind": "Wind output, MW",
    "solar": "Solar output, MW",
    "net_interchange": "Net interchange, MW (+ = export)",
    "hour": "Local hour",
    "month": "Local month",
}
FAMILY_SHORT = {
    "lead": "lead time",
    "calendar": "calendar at the target",
    "calendar_yesterday": "calendar 24 h before",
    "demand_now": "demand at the issue time",
    "demand_seasonal": "demand yesterday and last week",
    "weather": "weather at the target",
}


class BannedPhraseError(RuntimeError):
    """The generated report holds a banned causal phrase."""


@dataclass(slots=True)
class Results:
    sample: StoredSample
    shap: ShapSummary
    assoc: Associations
    tz: str
    seed: int


def compute(engine: Engine, cfg: AppConfig, region_id: str) -> Results:
    tz = cfg.region(region_id).local_tz
    sample = read_global(cfg, "dev")
    shap = summarise(sample, cfg.explain_report.shap, tz)
    first, last = period_bounds(cfg)
    data = load_association_data(engine, cfg, region_id, first, last)
    assoc = compute_associations(
        data, cfg.explain_report.associations, cfg.anomalies.seed, first, last
    )
    return Results(sample, shap, assoc, tz, cfg.anomalies.seed)


# --------------------------------------------------------------------------- formatting


def mw(v: float, digits: int = 0) -> str:
    if not np.isfinite(v):
        return "–"
    v = round(float(v), digits) + 0.0  # + 0.0 turns −0.0 into 0.0
    return ("−" if v < 0 else "") + f"{abs(v):,.{digits}f}"


def signed(v: float, digits: int = 0) -> str:
    if not np.isfinite(v):
        return "–"
    v = round(float(v), digits) + 0.0
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):,.{digits}f}"


def pct(v: float, digits: int = 1) -> str:
    return "–" if not np.isfinite(v) else f"{100 * v:.{digits}f}%"


def rate_ci(rate: float, lo: float, hi: float) -> str:
    if not np.isfinite(rate):
        return "–"
    return f"{100 * rate:.2f}% ({100 * lo:.2f}–{100 * hi:.2f})"


def fam_title(key: str) -> str:
    return FAMILY_SHORT.get(key, key)


# --------------------------------------------------------------------------- figures


def _style(ax: Axes, title: str, xlabel: str, ylabel: str, grid: Literal["x", "y"] = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=8)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.5)


def plot_families(shap: ShapSummary, path: Path) -> None:
    """Small multiples, one lead band per panel; each panel has its own MW scale."""
    fam = shap.families
    labels = [fam_title(k) for k in fam.keys]
    fig = Figure(figsize=(10, 6.4), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.2, right=0.96, top=0.86, bottom=0.08, hspace=0.45, wspace=0.75)
    fig.suptitle(
        "Mean |SHAP| by feature family and lead band (whole forecast, MW; own scale per panel)",
        color=INK,
        fontsize=11,
        x=0.03,
        ha="left",
    )
    y = np.arange(len(labels))[::-1]
    for ax, col in zip(axes.ravel(), shap.family_abs.columns, strict=True):
        v = shap.family_abs[col].to_numpy()
        ax.barh(y, v, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        top = float(v.max())
        for yi, vi in zip(y, v, strict=True):
            ax.text(vi + 0.02 * top, yi, mw(vi), va="center", fontsize=7, color=INK_2)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1.22 * top)
        _style(ax, f"Leads {col}", "mean |SHAP|, MW", "", grid="x")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def plot_weather(shap: ShapSummary, path: Path, quantiles: Sequence[float]) -> None:
    """Small multiples, one band per panel: mean weather contribution and its quantile range."""
    w = shap.weather[shap.weather["drawn"]]
    fig = Figure(figsize=(10, 6.4), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 2, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.1, hspace=0.35, wspace=0.22)
    lo, hi = (round(100 * q) for q in quantiles)
    fig.suptitle(
        "Weather family's contribution against the GFS forecast temperature at the target "
        f"(mean, {lo}–{hi}% range shaded)",
        color=INK,
        fontsize=11,
        x=0.03,
        ha="left",
    )
    for ax, band in zip(axes.ravel(), [band_label(b) for b in shap.bands], strict=True):
        sel = w[w["band"] == band]
        x = sel["lower"].to_numpy() + 0.5
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.fill_between(x, sel["lo"], sel["hi"], color=BAND_FILL, linewidth=0, zorder=2)
        ax.plot(x, sel["mean"], color=BLUE, linewidth=2, zorder=3)
        _style(ax, f"Leads {band}", "forecast temperature, °C", "contribution, MW")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def plot_rates(
    assoc: Associations, rate: str, path: Path, title: str, color: str, few_days: int
) -> None:
    """Small multiples, one observable per panel: the rate per bin with its interval. A bin
    seen on fewer than ``few_days`` NEM days is drawn hollow (its interval is unreliable)."""
    fig = Figure(figsize=(11, 6.8), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 4)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, hspace=0.95, wspace=0.3)
    fig.suptitle(title, color=INK, fontsize=11, x=0.03, ha="left")
    overall = float(assoc.overall[rate].iloc[0])
    for ax, var in zip(axes.ravel(), VARIABLES, strict=False):
        t = assoc.tables[var].frame
        x = np.arange(len(t))
        r = 100 * t[rate].to_numpy()
        err = np.vstack([r - 100 * t[f"{rate}_lo"].to_numpy(), 100 * t[f"{rate}_hi"] - r])
        ax.axhline(100 * overall, color=AXIS, linewidth=1, zorder=1)
        ax.errorbar(x, r, yerr=err, fmt="none", ecolor=color, elinewidth=1.2, zorder=2)
        few = t["days"].to_numpy() < few_days
        ax.plot(
            x[~few],
            r[~few],
            "o",
            color=color,
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            zorder=3,
        )
        ax.plot(x[few], r[few], "o", color=SURFACE, markersize=4.5, markeredgecolor=color, zorder=3)
        labels = list(t["bin_short"])
        if var in ("hour", "month"):
            step = 3 if var == "hour" else 1
            ax.set_xticks(x[::step], labels[::step])
        else:
            ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=6)
        ax.set_ylim(bottom=0)
        _style(ax, SHORT_TITLES[var], "", "% of intervals")
    axes.ravel()[-1].set_visible(False)
    fig.text(
        0.03,
        0.925,
        f"Grey line: the overall rate. Hollow: a bin seen on fewer than {few_days} NEM days. "
        "Bin labels are the edges (deciles of development values; temperature in fixed 5 °C bins).",
        color=MUTED,
        fontsize=7.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


# --------------------------------------------------------------------------- sections


def _gate_section(res: Results) -> tuple[str, bool]:
    s = res.shap
    rows = []
    by_band = {b: [f for f in res.sample.fits if tuple(f["band"]) == b] for b in s.bands}
    for b, fits in by_band.items():
        g = [f["gate"] for f in fits]
        rows.append(
            [
                band_label(b),
                str(len(fits)),
                f"{sum(x['n'] for x in g):,}",
                str(sum(x["forecast_mismatches"] for x in g)),
                f"{max(x['max_forecast_diff_mw'] for x in g):.1e}",
                str(sum(x["sum_failures"] for x in g)),
                f"{max(x['max_sum_error'] for x in g):.1e}",
            ]
        )
    total = s.gate
    passed = bool(total["passed"])
    rows.append(
        [
            "**all**",
            str(len(res.sample.fits)),
            f"{total['n']:,}",
            str(total["forecast_mismatches"]),
            f"{total['max_forecast_diff_mw']:.1e}",
            str(total["sum_failures"]),
            f"{total['max_sum_error']:.1e}",
        ]
    )
    table = _md_table(
        [
            "Leads",
            "Fits",
            "Pairs",
            "Forecast mismatches",
            "Largest \\|float64 − stored\\|, MW",
            "SHAP-sum failures",
            "Largest SHAP-sum error",
        ],
        rows,
    )
    verdict = "**PASS**" if passed else "**FAIL** (explanations withheld)"
    text = (
        "E1 (§3) asks that every explained pair's recomputed forecast, cast to float32 as M2 "
        "stored it, equals the stored forecast bit for bit, and that base value + Σ SHAP equals "
        f"the raw prediction within {total['tolerance']:g}. On the global sample: {verdict}. "
        "The float64 column is for information: the gate compares float32 values, and one "
        "float32 step near 8,000 MW is about 0.001 MW.\n\n" + table
    )
    return text, passed


def _cell(frame: pd.DataFrame, row: str, col: str) -> float:
    return float(frame[col].to_numpy()[list(frame.index).index(row)])


def _family_section(res: Results, fig_rel: str) -> str:
    s = res.shap
    bands = list(s.family_abs.columns)
    fam_rows = [
        [fam_title(k), *[mw(_cell(s.family_abs, k, b)) for b in bands]] for k in s.families.keys
    ]
    pairs_row = ["*pairs*", *[f"{s.pairs_by_band[b]:,}" for b in s.bands]]
    fam_table = _md_table(["Family (mean \\|SHAP\\|, MW)", *bands], [*fam_rows, pairs_row])
    gain_rows = [
        [fam_title(k), *[pct(_cell(s.gain_family, k, b)) for b in bands]] for k in s.families.keys
    ]
    gain_table = _md_table(["Family (share of split gain)", *bands], gain_rows)
    feat = s.feature_abs
    feat_rows = [
        [f"`{c}`", fam_title(s.families.of(c)), *[mw(_cell(feat, c, b), 1) for b in bands]]
        for c in feat.index
    ]
    feat_table = _md_table(["Feature", "Family", *bands], feat_rows, "ll" + "r" * len(bands))
    gf = s.gain_feature
    gf_rows = [[f"`{c}`", *[pct(_cell(gf, c, b)) for b in bands]] for c in gf.index]
    gf_table = _md_table(["Feature (share of split gain)", *bands], gf_rows)
    bf = s.family_abs_by_form
    bf_rows = [
        [
            r["band"],
            r["form"],
            str(r["fits"]),
            f"{r['pairs']:,}",
            *[mw(r[k]) for k in s.families.keys],
        ]
        for _, r in bf.iterrows()
    ]
    bf_table = _md_table(
        ["Leads", "Target form", "Fits", "Pairs", *[fam_title(k) for k in s.families.keys]],
        bf_rows,
        "ll" + "r" * (2 + len(s.families.keys)),
    )
    form_rows = []
    for b in s.bands:
        counts = s.fits[s.fits["band_start"] == b[0]]["form"].value_counts()
        form_rows.append([band_label(b), ", ".join(f"{k} {v}" for k, v in counts.items())])
    form_table = _md_table(["Leads", "Target forms of run 2's fits"], form_rows, "ll")
    top = [
        f"{b}: {fam_title(str(s.family_abs[b].idxmax()))} ({mw(s.family_abs[b].max())} MW)"
        for b in bands
    ]
    return "\n\n".join(
        [
            "Mean |SHAP| is the average size of a family's contribution to the forecast, in MW, "
            "on the whole-forecast decomposition. The largest family per band: "
            + "; ".join(top)
            + ".",
            f"![Mean |SHAP| by family and band]({fig_rel})",
            fam_table,
            "Run 2 chose a target form per retrain and band (M2 step 3):",
            form_table,
            "**M2's split gain** (tree-only, so it describes the correction to each fit's "
            "anchor, not the whole forecast), as a share of each fit's total, averaged over the "
            "band's fits:",
            gain_table,
            "**Raw TreeSHAP by target form** (tree-only, relative to each fit's anchor; mean "
            "\\|SHAP\\| of the family, MW). These are the values the headline adds the anchor to:",
            bf_table,
            "<details><summary>Every feature: mean |SHAP| (whole forecast, MW) and split-gain "
            "share</summary>\n\n" + feat_table + "\n\n" + gf_table + "\n\n</details>",
        ]
    )


def _stability_section(res: Results) -> str:
    st = res.shap.stability
    rows = [
        [
            r["band"],
            f"{r['retrain_a']} → {r['retrain_b']}",
            f"{r['form_a']} → {r['form_b']}",
            f"{r['spearman']:.2f}",
        ]
        for _, r in st.iterrows()
    ]
    summary_rows = []
    for band, grp in st.groupby("band", sort=False):
        same = grp[grp["form_a"] == grp["form_b"]]["spearman"]
        diff = grp[grp["form_a"] != grp["form_b"]]["spearman"]
        summary_rows.append(
            [
                str(band),
                f"{grp['spearman'].median():.2f}",
                f"{grp['spearman'].min():.2f}",
                f"{len(same)}: {same.median():.2f}" if len(same) else "–",
                f"{len(diff)}: {diff.median():.2f}" if len(diff) else "–",
            ]
        )
    return "\n\n".join(
        [
            "Spearman correlation between consecutive retrains' rankings of the 26 features by "
            "mean |SHAP| (whole forecast), per band. 1 means the same order.",
            _md_table(
                [
                    "Leads",
                    "Median",
                    "Lowest",
                    "Same form: pairs, median",
                    "Form changed: pairs, median",
                ],
                summary_rows,
            ),
            "<details><summary>Every consecutive pair</summary>\n\n"
            + _md_table(["Leads", "Retrains", "Target forms", "Spearman"], rows, "lllr")
            + "\n\n</details>",
        ]
    )


def _weather_section(res: Results, fig_rel: str, cfg: AppConfig) -> str:
    sc = cfg.explain_report.shap
    w = res.shap.weather
    q5 = band_label((sc.question5_band[0], sc.question5_band[1]))
    sel = w[(w["band"] == q5) & w["drawn"]]
    rows = [
        [
            f"{r['lower']:.0f} to {r['lower'] + sc.weather_bin_c:.0f}",
            f"{int(r['n']):,}",
            signed(r["mean"]),
            f"{signed(r['lo'])} to {signed(r['hi'])}",
        ]
        for _, r in sel.iterrows()
    ]
    lo, hi = (round(100 * q) for q in sc.weather_range)
    return "\n\n".join(
        [
            "The weather family's contribution (the sum of its five features' SHAP values) "
            "against `wx_temp`, the population-weighted GFS forecast temperature at the target, "
            f"in {sc.weather_bin_c:g} °C bins. Bins with fewer than {sc.weather_min_pairs} pairs "
            "are not drawn. None of the weather features is part of an anchor, so these values "
            "are the trees' own.",
            f"![Weather contribution against forecast temperature]({fig_rel})",
            f"<details><summary>Leads {q5}, every drawn bin</summary>\n\n"
            + _md_table(
                ["Forecast temperature, °C (bin)", "Pairs", "Mean, MW", f"{lo}–{hi}% range, MW"],
                rows,
            )
            + "\n\n</details>",
        ]
    )


def _question5_section(res: Results, cfg: AppConfig) -> str:
    sc = cfg.explain_report.shap
    q = res.shap.question5
    groups = list(dict.fromkeys(q["group"]))
    terms = list(dict.fromkeys(q["term"]))
    names = {"constant": "constant (base value + sample mean of the anchor)"}
    names |= {k: fam_title(k) for k in res.shap.families.keys}
    names["forecast_total"] = "**forecast**"
    parts = []
    for basis, title in (("actual", "actual demand"), ("forecast", "forecast demand")):
        sub = q[q["basis"] == basis]
        head = []
        for g in groups:
            r = sub[sub["group"] == g].iloc[0]
            head.append(
                f"{g}: above {mw(r['threshold'])} MW, {int(r['n_top']):,} of "
                f"{int(r['n_all']):,} pairs"
            )
        rows = []
        for t in terms:
            row = [names[t]]
            for g in groups:
                r = sub[(sub["group"] == g) & (sub["term"] == t)].iloc[0]
                row += [mw(r["top"]), mw(r["all"]), signed(r["diff"])]
            rows.append(row)
        headers = ["Term (mean, MW)"]
        for g in groups:
            headers += [f"{g}: top {sc.top_share:.0%}", f"{g}: all", f"{g}: difference"]
        parts.append(
            f"**Top {sc.top_share:.0%} of {title}** ({'; '.join(head)}):\n\n"
            + _md_table(headers, rows)
        )
    lead = []
    sub = q[(q["basis"] == "actual") & q["term"].isin(res.shap.families.keys)]
    for g in groups:
        r = sub[sub["group"] == g].sort_values("diff", key=np.abs, ascending=False).iloc[0]
        lead.append(f"{g}: {fam_title(r['term'])} ({signed(r['diff'])} MW)")
    q5_label = band_label((sc.question5_band[0], sc.question5_band[1]))
    seasons = "; ".join(f"{k}: months {', '.join(map(str, v))}" for k, v in sc.seasons.items())
    return "\n\n".join(
        [
            f"In the {q5_label} band, the pooled global sample of its "
            f"{int((res.shap.fits['band_start'] == sc.question5_band[0]).sum())} fits. Each "
            f"row's top {sc.top_share:.0%} is taken within its own group ({seasons}; by the "
            "target's local month). The terms add up to the forecast, so each difference column "
            "sums to the forecast's difference. They describe the model's arithmetic on a sample, "
            "not the grid's physics, and no interval is attached. The constant also differs "
            "between the top 5% and all targets: each fit has its own constant, and the top 5% "
            "are not spread evenly over the fits.",
            "The family that differs most between the top 5% of actual demand and all targets: "
            + "; ".join(lead)
            + ".",
            *parts,
        ]
    )


def _rate_rows(t: pd.DataFrame) -> list[list[str]]:
    return [
        [
            str(r["bin"]),
            f"{int(r['n']):,}",
            f"{int(r['days']):,}",
            pct(r["share"]),
            rate_ci(r["high"], r["high_lo"], r["high_hi"]),
            rate_ci(r["low"], r["low_lo"], r["low_hi"]),
        ]
        for _, r in t.iterrows()
    ]


def _extreme(r: pd.Series, rate: str) -> str:
    days = int(r["days"])
    return (
        f"{pct(r[rate], 2)} ({r['bin']}; {int(r['n']):,} intervals on {days:,} "
        f"day{'s' if days != 1 else ''})"
    )


def _range_rows(assoc: Associations) -> list[list[str]]:
    rows = []
    for v in VARIABLES:
        t = assoc.tables[v].frame
        t = t[t["n"] > 0]
        cells = [TITLES[v]]
        for rate in ("high", "low"):
            r = t[rate].to_numpy()
            lo, hi = t.iloc[int(np.argmin(r))], t.iloc[int(np.argmax(r))]
            cells += [_extreme(lo, rate), _extreme(hi, rate)]
        rows.append(cells)
    return rows


def _associations_section(res: Results, cfg: AppConfig, figs: dict[str, str]) -> str:
    a = res.assoc
    ac = cfg.explain_report.associations
    o = a.overall.iloc[0]
    hp, lp = f"${ac.high_price:,.0f}", f"${ac.low_price:,.0f}"
    lvl = f"{ac.ci_level:.0%}"
    out = [
        f"All 5-minute intervals of NEM days {a.first} → {a.last} ({a.n:,} intervals, {a.days:,} "
        f"days). Overall, P(price > {hp}/MWh) = {rate_ci(o['high'], o['high_lo'], o['high_hi'])} "
        f"and P(price < {lp}/MWh) = {rate_ci(o['low'], o['low_lo'], o['low_hi'])} ({lvl} "
        f"intervals from {a.resamples:,} day-block bootstrap resamples, NEM days, seeded). Every "
        "rate is a share of intervals in the same 5 minutes as the observed condition: an "
        "association only. A rate that is higher in a bin means high (or negative) prices were "
        "more frequent when that condition held, nothing more.",
        "**Bins.** Deciles are right-closed bins between the development 10% … 90% quantiles "
        "(the bin labels are the edges). Solar output is exactly 0 MW in "
        f"{pct(a.tables['solar'].frame['share'].iloc[0])} of intervals, so it has a zero-output "
        "bin plus deciles of positive output. Temperature bins are fixed 5 °C bins that include "
        "their lower edge. These edges are fixed now: the holdout (step 6) reuses them. "
        "*Days* counts the NEM days with at least one interval in the bin: a bin seen on few "
        "days has an unreliable interval (a bin seen on one day gets a zero-width one).",
        "**Lowest and highest rate across each variable's bins:**",
        _md_table(
            [
                "Observable",
                f"P(> {hp}): lowest",
                f"P(> {hp}): highest",
                f"P(< {lp}): lowest",
                f"P(< {lp}): highest",
            ],
            _range_rows(a),
            "lllll",
        ),
        f"![P(price > {hp}) by observable]({figs['high']})",
        f"![P(price < {lp}) by observable]({figs['low']})",
    ]
    for v in VARIABLES:
        t = a.tables[v]
        note = f" {t.missing:,} intervals lack the value and are left out." if t.missing else ""
        out.append(
            f"<details><summary>{TITLES[v]}</summary>\n\n"
            + (note.strip() + "\n\n" if note else "")
            + _md_table(
                ["Bin", "Intervals", "Days", "Share", f"P(> {hp}) ({lvl})", f"P(< {lp}) ({lvl})"],
                _rate_rows(t.frame),
            )
            + "\n\n</details>"
        )
    tw = a.two_way
    tw_rows = [
        [
            f"{r['demand']} demand",
            f"{r['wind']} wind",
            f"{int(r['n']):,}",
            f"{int(r['days']):,}",
            rate_ci(r["high"], r["high_lo"], r["high_hi"]),
            rate_ci(r["low"], r["low_lo"], r["low_hi"]),
        ]
        for _, r in tw.iterrows()
    ]
    d_edges = ", ".join(f"{e:,.0f}" for e in a.binnings["demand_two_way"].edges)
    w_edges = ", ".join(f"{e:,.0f}" for e in a.binnings["wind_two_way"].edges)
    out += [
        f"**Demand tercile × wind tercile** (demand edges {d_edges} MW; wind edges {w_edges} MW):",
        _md_table(
            ["Demand", "Wind", "Intervals", "Days", f"P(> {hp}) ({lvl})", f"P(< {lp}) ({lvl})"],
            tw_rows,
            "llrrrr",
        ),
    ]
    return "\n\n".join(out)


def nem_days_of(period: Sequence[str]) -> tuple[str, str]:
    """First and last NEM day of a [start, end) period of UTC issue times."""
    start = pd.Timestamp(period[0]).tz_convert(NEM_TZ)
    end = pd.Timestamp(period[1]).tz_convert(NEM_TZ) - pd.Timedelta(days=1)
    return start.date().isoformat(), end.date().isoformat()


def _provenance(res: Results, cfg: AppConfig) -> str:
    m = res.sample.manifest
    first, last = nem_days_of(m["period"])
    return (
        f"Generated by `gridcast report-m4-explain` at commit `{(git_sha() or 'unknown')[:7]}` "
        f"from the stored global SHAP sample (`gridcast explain-dev` at commit "
        f"`{m['commit'][:7]}`, {m['created_at'][:16].replace('T', ' ')} UTC: run 2, batch "
        f"`{m['batch_id'][:8]}…`, {len(m['fits'])} fits, up to {m['sample_pairs']:,} seeded "
        f"pairs per fit over the issue times of NEM days {first} → {last}) and "
        "the database. Every number below is computed from those outputs. The decisions behind "
        'it are in [`docs/plans/m4_plan.md`](../docs/plans/m4_plan.md) §1 ("Clarified in step '
        '5"), §2.6 and §2.7. Reported only: E1 is the one criterion here.'
    )


def _how_to_read() -> str:
    return "\n".join(
        [
            "- **Whole forecast.** Each M2 fit predicts a correction to an anchor that depends on "
            "its target form: nothing (*direct*), the latest demand (*latest*), demand at the "
            "same time yesterday (*residual*), or both plus yesterday's change (*anchored*). Raw "
            "TreeSHAP covers only the correction. The headline adds each fit's anchor to the "
            "features it is made of, as x − x̄ over the fit's own sample, which is the exact "
            "Shapley value of a linear term. Every pair then reads forecast = constant + Σ "
            "contributions, exactly, whatever the form, so fits of different forms can be pooled.",
            "- **The mix is disclosed.** The trees' part is path-dependent TreeSHAP and the "
            "anchor's part is an interventional attribution against the sample, so the sum is "
            "the Shapley value of neither game exactly. It adds up to the forecast. The raw "
            "tree-only values are shown by target form beside the headline.",
            "- **The model's arithmetic, not the grid's physics.** A contribution says how the "
            "model's forecast moved with an input. Correlated inputs (several temperature terms, "
            "demand lags) share credit among themselves, so families are the main unit.",
            "- **Price associations are association only:** rates by same-interval conditions, "
            "with no model and no claim about mechanism.",
        ]
    )


def _disclosures(cfg: AppConfig) -> str:
    return "\n".join(
        [
            "- These analyses were designed after M2's and M3's development and holdout results "
            "had been seen (plan §3.3). Their settings were fixed in the step-5 question round, "
            "before any global SHAP value or association rate was computed.",
            "- While this report's code was being written, the progress log of `explain-dev` "
            "was read. It shows each fit's E1 result, pair count and time, and no SHAP value.",
            "- ERA5 temperature is retrospective (published days later). It describes the same "
            "interval as the price, which is what an association needs, but it was not known "
            "at the time.",
            "- The day-block bootstrap resamples NEM days, so it keeps the dependence within a "
            "day but not across days (multi-day heat or wind droughts). The intervals may be "
            "too narrow where such spells dominate a bin.",
            "- Two presentation details were added after the first draft of this report was "
            "read, and change no number: the *Days* column with the hollow markers for bins seen "
            f"on fewer than {cfg.explain_report.associations.few_days} NEM days (the first draft "
            "showed a 2-interval temperature bin at 100%), and the cleaner labels.",
            "- The global sample is 20,000 seeded pairs per fit (fewer where fewer were eligible, "
            "e.g. the GFS gap of Dec 2023 → Jan 2024), pooled with one weight per pair.",
        ]
    )


def render(res: Results, cfg: AppConfig, figs: dict[str, str]) -> tuple[str, bool]:
    gate_text, passed = _gate_section(res)
    sections = [
        "# M4 explanations: the demand model's SHAP values and price associations "
        "(development, Victoria)",
        _provenance(res, cfg),
        "## How to read this report",
        _how_to_read(),
        "## 1. E1 on the global sample",
        gate_text,
    ]
    if passed:
        sections += [
            "## 2. Which inputs the forecasts lean on",
            _family_section(res, figs["families"]),
            "## 3. Stability across retrains",
            _stability_section(res),
            "## 4. The weather contribution against forecast temperature",
            _weather_section(res, figs["weather"], cfg),
            "## 5. Question 5, demand: what the model adds for high-demand targets",
            _question5_section(res, cfg),
        ]
    else:
        sections.append("E1 failed, so every SHAP summary is withheld (§2.6).")
    sections += [
        "## 6. Question 5, price: when high and negative prices were more frequent",
        _associations_section(res, cfg, figs),
        "## Disclosures",
        _disclosures(cfg),
    ]
    return "\n\n".join(sections) + "\n", passed


def write_report(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    res = compute(engine, cfg, region_id)
    reports = cfg.resolve(cfg.paths.reports_dir)
    fig_dir = reports / "figures"
    figs = {k: f"figures/{v}" for k, v in FIGURES.items()}
    text_, passed = render(res, cfg, figs)
    found = banned_in(text_, cfg.anomalies.evidence.banned_phrases)
    if found:
        raise BannedPhraseError(f"the report holds banned phrases: {found}")
    ac = cfg.explain_report.associations
    if passed:
        plot_families(res.shap, fig_dir / FIGURES["families"])
        plot_weather(res.shap, fig_dir / FIGURES["weather"], cfg.explain_report.shap.weather_range)
    plot_rates(
        res.assoc,
        "high",
        fig_dir / FIGURES["high"],
        f"P(price > ${ac.high_price:,.0f}/MWh) by same-interval observable "
        f"(development, {ac.ci_level:.0%} day-block intervals)",
        ORANGE,
        ac.few_days,
    )
    plot_rates(
        res.assoc,
        "low",
        fig_dir / FIGURES["low"],
        f"P(price < ${ac.low_price:,.0f}/MWh) by same-interval observable "
        f"(development, {ac.ci_level:.0%} day-block intervals)",
        BLUE,
        ac.few_days,
    )
    path = reports / REPORT_NAME
    path.write_text(text_, encoding="utf-8")
    edges = {
        k: {"kind": b.kind, "edges": list(b.edges), "width": b.width}
        for k, b in res.assoc.binnings.items()
    }
    edge_path = res.sample.directory.parent / EDGES_FILE
    first, last = res.assoc.first.isoformat(), res.assoc.last.isoformat()
    edge_path.write_text(json.dumps({"period": [first, last], **edges}, indent=1) + "\n")
    return path

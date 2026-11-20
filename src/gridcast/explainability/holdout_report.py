"""Generate ``reports/m4_explainability_holdout.md`` (+ figures) for M4 step 6.

Step 5's tables on the holdout, each with the development value beside it (m4_plan.md §1,
"Clarified in step 6"):

* **Global SHAP** of ``holdout_main``'s fits from ``gridcast explain-holdout`` (holdout issue
  times only), on step 5's whole-forecast decomposition;
* **Question 5, demand:** SHAP by family for high-demand holdout targets, with the holdout's
  seasons (summer Dec 2025 – Feb 2026, winter Jun – Aug 2026);
* **Question 5, price:** the rates on NEM days ``holdout_start`` → ``holdout_end`` on the
  stored development bin edges (checked against a recomputation from development data first).

Every number comes from the stored samples or the database. Reported only: E1 is the one
criterion here. The report refuses its own text if it holds a banned causal phrase.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.anomalies.evidence import banned_in
from gridcast.config import AppConfig, PriceAssociationsConfig
from gridcast.evaluation.report import _md_table
from gridcast.explainability.associations import (
    TITLES,
    VARIABLES,
    Associations,
    Binning,
    compute_associations,
    development_binnings,
    load_association_data,
    period_bounds,
)
from gridcast.explainability.global_run import read_global
from gridcast.explainability.report import (
    BLUE,
    EDGES_FILE,
    ORANGE,
    BannedPhraseError,
    Results,
    _cell,
    _gate_section,
    _how_to_read,
    _question5_section,
    _range_rows,
    fam_title,
    mw,
    nem_days_of,
    pct,
    plot_families,
    plot_rates,
    plot_weather,
    rate_ci,
    signed,
)
from gridcast.explainability.summary import band_label, summarise
from gridcast.provenance import git_sha

REPORT_NAME = "m4_explainability_holdout.md"
FIGURES = {
    "families": "m4h_explain_families_by_band.png",
    "weather": "m4h_explain_weather_dependence.png",
    "high": "m4h_explain_price_above_300.png",
    "low": "m4h_explain_price_negative.png",
}
CATEGORIES = {"hour": tuple(range(24)), "month": tuple(range(1, 13))}


class EdgesMismatch(RuntimeError):
    """The stored development bin edges differ from a recomputation."""


# --------------------------------------------------------------------------- the bins


def stored_binnings(path: Path) -> dict[str, Binning]:
    """The development bins step 5 stored (``association_edges_dev.json``)."""
    raw = json.loads(path.read_text())
    out = {}
    for key, spec in raw.items():
        if key == "period":
            continue
        variable = key.removesuffix("_two_way")
        out[key] = Binning(
            variable,
            spec["kind"],
            tuple(float(e) for e in spec["edges"]),
            float(spec["width"]),
            CATEGORIES.get(variable, ()) if spec["kind"] == "category" else (),
        )
    return out


def check_binnings(stored: dict[str, Binning], recomputed: dict[str, Binning]) -> None:
    if stored.keys() != recomputed.keys():
        raise EdgesMismatch(f"variables {sorted(stored)} != {sorted(recomputed)}")
    differ = [
        k for k in stored if dataclasses.astuple(stored[k]) != dataclasses.astuple(recomputed[k])
    ]
    if differ:
        raise EdgesMismatch(f"stored bin edges differ from development data: {', '.join(differ)}")


# --------------------------------------------------------------------------- compute


@dataclass(slots=True)
class HoldoutResults:
    hold: Results
    dev: Results


def holdout_bounds(cfg: AppConfig) -> tuple[dt.date, dt.date]:
    end = cfg.backtest.holdout_end
    if end is None:
        raise ValueError("backtest.holdout_end is not set")
    return cfg.backtest.holdout_start, end - dt.timedelta(days=1)


def compute(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    stand_in: bool = False,
) -> HoldoutResults:
    """Holdout and development results. ``stand_in`` renders the development sample and
    period in the holdout's place (a code test that reads no holdout value)."""
    tz = cfg.region(region_id).local_tz
    seed = cfg.anomalies.seed
    sc, ac = cfg.explain_report.shap, cfg.explain_report.associations
    dev_sample = read_global(cfg, "dev")
    first_d, last_d = period_bounds(cfg)
    data_d = load_association_data(engine, cfg, region_id, first_d, last_d)
    bins = stored_binnings(dev_sample.directory.parent / EDGES_FILE)
    check_binnings(bins, development_binnings(data_d, ac))
    assoc_d = compute_associations(data_d, ac, seed, first_d, last_d, bins)
    dev = Results(dev_sample, summarise(dev_sample, sc, tz), assoc_d, tz, seed)
    if stand_in:
        return HoldoutResults(dev, dev)
    hold_sample = read_global(cfg, "holdout")
    first_h, last_h = holdout_bounds(cfg)
    data_h = load_association_data(engine, cfg, region_id, first_h, last_h)
    assoc_h = compute_associations(data_h, ac, seed, first_h, last_h, bins)
    hold = Results(hold_sample, summarise(hold_sample, sc, tz), assoc_h, tz, seed)
    return HoldoutResults(hold, dev)


# --------------------------------------------------------------------------- sections


def beside(h: str, d: str) -> str:
    return f"{h} ({d})"


def _family_section(res: HoldoutResults, fig_rel: str) -> str:
    s, sd = res.hold.shap, res.dev.shap
    bands = list(s.family_abs.columns)
    rows = [
        [fam_title(k)]
        + [beside(mw(_cell(s.family_abs, k, b)), mw(_cell(sd.family_abs, k, b))) for b in bands]
        for k in s.families.keys
    ]
    rows.append(
        ["*pairs*"]
        + [beside(f"{s.pairs_by_band[b]:,}", f"{sd.pairs_by_band[b]:,}") for b in s.bands]
    )
    gain_rows = [
        [fam_title(k)]
        + [beside(pct(_cell(s.gain_family, k, b)), pct(_cell(sd.gain_family, k, b))) for b in bands]
        for k in s.families.keys
    ]
    form_rows = []
    for b in s.bands:
        counts = s.fits[s.fits["band_start"] == b[0]]["form"].value_counts()
        dcounts = sd.fits[sd.fits["band_start"] == b[0]]["form"].value_counts()
        form_rows.append(
            [
                band_label(b),
                ", ".join(f"{k} {v}" for k, v in counts.items()),
                ", ".join(f"{k} {v}" for k, v in dcounts.items()),
            ]
        )
    bf = s.family_abs_by_form
    bf_rows = [
        [r["band"], r["form"], str(r["fits"]), f"{r['pairs']:,}"]
        + [mw(r[k]) for k in s.families.keys]
        for _, r in bf.iterrows()
    ]
    feat = s.feature_abs
    feat_rows = [
        [f"`{c}`", fam_title(s.families.of(c))]
        + [beside(mw(_cell(feat, c, b), 1), mw(_cell(sd.feature_abs, c, b), 1)) for b in bands]
        for c in feat.index
    ]
    top = [
        f"{b}: {fam_title(str(s.family_abs[b].idxmax()))} ({mw(s.family_abs[b].max())} MW; "
        f"in development: {fam_title(str(sd.family_abs[b].idxmax()))})"
        for b in bands
    ]
    return "\n\n".join(
        [
            "Mean |SHAP| of each family on the whole-forecast decomposition, in MW, holdout "
            "first and development in brackets. The largest family per band on the holdout: "
            + "; ".join(top)
            + ".",
            f"![Mean |SHAP| by family and band, holdout]({fig_rel})",
            _md_table(["Family (mean \\|SHAP\\|, MW): holdout (development)", *bands], rows),
            "`holdout_main` chose its target forms by the frozen M2 recipe at each retrain, "
            "as run 2 did:",
            _md_table(["Leads", "Holdout fits", "Development fits (run 2)"], form_rows, "lll"),
            "**M2's split gain** (tree-only), as a share of each fit's total, averaged over the "
            "band's fits, holdout (development):",
            _md_table(["Family (share of split gain)", *bands], gain_rows),
            "**Raw TreeSHAP by target form** on the holdout (tree-only, relative to each fit's "
            "anchor; mean \\|SHAP\\| of the family, MW):",
            _md_table(
                ["Leads", "Target form", "Fits", "Pairs", *[fam_title(k) for k in s.families.keys]],
                bf_rows,
                "ll" + "r" * (2 + len(s.families.keys)),
            ),
            "<details><summary>Every feature: mean |SHAP| (whole forecast, MW), holdout "
            "(development)</summary>\n\n"
            + _md_table(["Feature", "Family", *bands], feat_rows, "ll" + "r" * len(bands))
            + "\n\n</details>",
        ]
    )


def _stability_section(res: HoldoutResults) -> str:
    st, sd = res.hold.shap.stability, res.dev.shap.stability
    dev_median = sd.groupby("band", sort=False)["spearman"].median()
    rows = []
    for band, grp in st.groupby("band", sort=False):
        same = grp[grp["form_a"] == grp["form_b"]]["spearman"]
        diff = grp[grp["form_a"] != grp["form_b"]]["spearman"]
        rows.append(
            [
                str(band),
                str(len(grp)),
                f"{grp['spearman'].median():.2f}",
                f"{grp['spearman'].min():.2f}",
                f"{len(same)}: {same.median():.2f}" if len(same) else "–",
                f"{len(diff)}: {diff.median():.2f}" if len(diff) else "–",
                f"{dev_median.get(band, np.nan):.2f}",
            ]
        )
    every = [
        [
            r["band"],
            f"{r['retrain_a']} → {r['retrain_b']}",
            f"{r['form_a']} → {r['form_b']}",
            f"{r['spearman']:.2f}",
        ]
        for _, r in st.iterrows()
    ]
    return "\n\n".join(
        [
            "Spearman correlation between consecutive holdout retrains' rankings of the 26 "
            "features by mean |SHAP| (whole forecast), per band. 1 means the same order.",
            _md_table(
                [
                    "Leads",
                    "Pairs",
                    "Median",
                    "Lowest",
                    "Same form: pairs, median",
                    "Form changed: pairs, median",
                    "Development median",
                ],
                rows,
            ),
            "<details><summary>Every consecutive holdout pair</summary>\n\n"
            + _md_table(["Leads", "Retrains", "Target forms", "Spearman"], every, "lllr")
            + "\n\n</details>",
        ]
    )


def _weather_section(res: HoldoutResults, fig_rel: str, cfg: AppConfig) -> str:
    sc = cfg.explain_report.shap
    q5 = band_label((sc.question5_band[0], sc.question5_band[1]))
    w = res.hold.shap.weather
    wd = res.dev.shap.weather
    sel = w[(w["band"] == q5) & w["drawn"]]
    dev = wd[(wd["band"] == q5) & wd["drawn"]].set_index("lower")
    rows = [
        [
            f"{r['lower']:.0f} to {r['lower'] + sc.weather_bin_c:.0f}",
            f"{int(r['n']):,}",
            signed(r["mean"]),
            f"{signed(r['lo'])} to {signed(r['hi'])}",
            signed(float(dev.loc[r["lower"], "mean"])) if r["lower"] in dev.index else "–",
        ]
        for _, r in sel.iterrows()
    ]
    lo, hi = (round(100 * q) for q in sc.weather_range)
    return "\n\n".join(
        [
            "The weather family's contribution against `wx_temp`, the population-weighted GFS "
            f"forecast temperature at the target, in {sc.weather_bin_c:g} °C bins (fewer than "
            f"{sc.weather_min_pairs} pairs: not drawn). None of the weather features is part of "
            "an anchor, so these values are the trees' own.",
            f"![Weather contribution against forecast temperature, holdout]({fig_rel})",
            f"Leads {q5}, every drawn holdout bin, with the development mean beside it:",
            _md_table(
                [
                    "Forecast temperature, °C (bin)",
                    "Pairs",
                    "Mean, MW",
                    f"{lo}–{hi}% range, MW",
                    "Development mean, MW",
                ],
                rows,
            ),
        ]
    )


def _question5_beside(res: HoldoutResults, cfg: AppConfig) -> str:
    q, qd = res.hold.shap.question5, res.dev.shap.question5
    keys = list(res.hold.shap.families.keys)
    groups = list(dict.fromkeys(q["group"]))
    rows = []
    for t in [*keys, "forecast_total"]:
        name = "**forecast**" if t == "forecast_total" else fam_title(t)
        row = [name]
        for g in groups:
            h = q[(q["basis"] == "actual") & (q["group"] == g) & (q["term"] == t)]
            d = qd[(qd["basis"] == "actual") & (qd["group"] == g) & (qd["term"] == t)]
            row.append(
                beside(
                    signed(float(h["diff"].iloc[0])) if len(h) else "–",
                    signed(float(d["diff"].iloc[0])) if len(d) else "–",
                )
            )
        rows.append(row)
    share = cfg.explain_report.shap.top_share
    return (
        f"**Top {share:.0%} of actual demand minus all targets, holdout (development), mean MW "
        "by family:**\n\n" + _md_table(["Family", *groups], rows)
    )


def _overall(a: Associations, ac: PriceAssociationsConfig) -> str:
    o = a.overall.iloc[0]
    return (
        f"P(> ${ac.high_price:,.0f}) {rate_ci(o['high'], o['high_lo'], o['high_hi'])}, "
        f"P(< ${ac.low_price:,.0f}) {rate_ci(o['low'], o['low_lo'], o['low_hi'])}"
    )


def _variable_rows(h: pd.DataFrame, d: pd.DataFrame) -> list[list[str]]:
    dev = d.set_index("bin")
    rows = []
    for _, r in h.iterrows():
        dr = dev.loc[r["bin"]] if r["bin"] in dev.index else None
        rows.append(
            [
                str(r["bin"]),
                f"{int(r['n']):,}",
                f"{int(r['days']):,}",
                pct(r["share"]),
                rate_ci(r["high"], r["high_lo"], r["high_hi"]),
                rate_ci(r["low"], r["low_lo"], r["low_hi"]),
                "–" if dr is None else pct(dr["share"]),
                "–" if dr is None else pct(dr["high"], 2),
                "–" if dr is None else pct(dr["low"], 2),
            ]
        )
    return rows


def _associations_section(res: HoldoutResults, cfg: AppConfig, figs: dict[str, str]) -> str:
    a, ad = res.hold.assoc, res.dev.assoc
    ac = cfg.explain_report.associations
    hp, lp = f"${ac.high_price:,.0f}", f"${ac.low_price:,.0f}"
    lvl = f"{ac.ci_level:.0%}"
    n_high = round(float(a.overall["high"].iloc[0]) * a.n)
    out = [
        f"All 5-minute intervals of NEM days {a.first} → {a.last} ({a.n:,} intervals, "
        f"{a.days:,} days): {_overall(a, ac)} ({lvl} intervals from {a.resamples:,} day-block "
        f"bootstrap resamples, NEM days, seeded). Development (NEM days {ad.first} → {ad.last}): "
        f"{_overall(ad, ac)}. Every rate is a share of intervals in the same 5 minutes as the "
        "observed condition: an association only.",
        "**Bins.** The bins are development's, read from step 5's stored edges and checked "
        "against a recomputation from development data. A shift in conditions between the "
        "periods shows as the *Share* of intervals per bin, beside development's. "
        f"{n_high:,} intervals of this period had a price above {hp}, so a bin that holds few "
        "of them has a wide or degenerate interval; every bin's counts are shown, and no bin "
        "is dropped. *Days* counts the NEM days with at least one interval in the bin: a bin "
        "seen on few days has an unreliable interval.",
        "**Lowest and highest holdout rate across each variable's bins:**",
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
        f"![P(price > {hp}) by observable, holdout]({figs['high']})",
        f"![P(price < {lp}) by observable, holdout]({figs['low']})",
    ]
    for v in VARIABLES:
        t = a.tables[v]
        note = f"{t.missing:,} intervals lack the value and are left out.\n\n" if t.missing else ""
        out.append(
            f"<details><summary>{TITLES[v]}</summary>\n\n"
            + note
            + _md_table(
                [
                    "Bin",
                    "Intervals",
                    "Days",
                    "Share",
                    f"P(> {hp}) ({lvl})",
                    f"P(< {lp}) ({lvl})",
                    "Development share",
                    f"Development P(> {hp})",
                    f"Development P(< {lp})",
                ],
                _variable_rows(t.frame, ad.tables[v].frame),
            )
            + "\n\n</details>"
        )
    dev_tw = ad.two_way.set_index(["demand", "wind"])
    tw_rows = []
    for _, r in a.two_way.iterrows():
        d = dev_tw.loc[(r["demand"], r["wind"])]
        tw_rows.append(
            [
                f"{r['demand']} demand",
                f"{r['wind']} wind",
                f"{int(r['n']):,}",
                f"{int(r['days']):,}",
                rate_ci(r["high"], r["high_lo"], r["high_hi"]),
                rate_ci(r["low"], r["low_lo"], r["low_hi"]),
                pct(d["high"], 2),
                pct(d["low"], 2),
            ]
        )
    d_edges = ", ".join(f"{e:,.0f}" for e in a.binnings["demand_two_way"].edges)
    w_edges = ", ".join(f"{e:,.0f}" for e in a.binnings["wind_two_way"].edges)
    out += [
        f"**Demand tercile × wind tercile** (development edges: demand {d_edges} MW; wind "
        f"{w_edges} MW):",
        _md_table(
            [
                "Demand",
                "Wind",
                "Intervals",
                "Days",
                f"P(> {hp}) ({lvl})",
                f"P(< {lp}) ({lvl})",
                f"Development P(> {hp})",
                f"Development P(< {lp})",
            ],
            tw_rows,
            "llrrrrrr",
        ),
    ]
    return "\n\n".join(out)


def _provenance(res: HoldoutResults) -> str:
    m, md = res.hold.sample.manifest, res.dev.sample.manifest
    first, last = nem_days_of(m["period"])
    runs = 1 + len(m.get("previous_runs", []))
    return (
        f"Generated by `gridcast report-m4-explain-holdout` at commit "
        f"`{(git_sha() or 'unknown')[:7]}` from the stored holdout SHAP sample (`gridcast "
        f"explain-holdout` at commit `{m['commit'][:7]}`, "
        f"{m['created_at'][:16].replace('T', ' ')} UTC; {runs} run{'s' if runs != 1 else ''}: "
        f"`holdout_main`, batch `{m['batch_id'][:8]}…`, {len(m['fits'])} fits, up to "
        f"{m['sample_pairs']:,} seeded pairs per fit over the issue times of NEM days {first} → "
        f"{last}), the development sample of step 5 (`explain-dev` at commit "
        f"`{md['commit'][:7]}`, {len(md['fits'])} fits) and the database. Every number below is "
        "computed from those outputs. The decisions behind it are in "
        '[`docs/plans/m4_plan.md`](../docs/plans/m4_plan.md) §1 ("Clarified in step 5" and '
        '"Clarified in step 6"), §2.6 and §2.7. Reported only: E1 is the one criterion here. '
        "Development values, from [`m4_explainability.md`](m4_explainability.md), are shown "
        "beside the holdout's, in brackets or in their own columns."
    )


def _disclosures(cfg: AppConfig) -> str:
    return "\n".join(
        [
            "- These analyses were designed after M2's and M3's development and holdout results "
            "had been seen. The planning saw some holdout price facts (plan §3.3): 0.2% of "
            "intervals above $300 and 25.9% negative. The settings were fixed in step 5 and the "
            "step-6 question round, before any holdout SHAP value or holdout association rate "
            "was computed.",
            "- `explain-holdout` printed only its E1 gates and timings. This report's code was "
            "committed before any holdout SHAP value or holdout association rate was read. It "
            "was tested by rendering it with the development sample and period standing in for "
            "the holdout's.",
            "- The holdout sample covers holdout issue times only. `holdout_main`'s first "
            "retrain (2025-08-01) contributes October 2025. Its August–September 2025 issue "
            "times belong to development, where run 2's identical model covers them (plan §1, "
            "step 3).",
            "- ERA5 temperature is retrospective (published days later). It describes the same "
            "interval as the price, which is what an association needs, but it was not known "
            "at the time.",
            "- The day-block bootstrap resamples NEM days, so it keeps the dependence within a "
            "day but not across days. The intervals may be too narrow where multi-day spells "
            "dominate a bin.",
            f"- Bins seen on fewer than {cfg.explain_report.associations.few_days} NEM days are "
            "drawn hollow in the figures, as in development.",
        ]
    )


def render(res: HoldoutResults, cfg: AppConfig, figs: dict[str, str]) -> tuple[str, bool]:
    gate_text, passed = _gate_section(res.hold)
    sc = cfg.explain_report.shap
    seasons = "; ".join(f"{k}: months {', '.join(map(str, v))}" for k, v in sc.seasons.items())
    sections = [
        "# M4 explanations on the holdout year: the demand model's SHAP values and price "
        "associations (Victoria)",
        _provenance(res),
        "## How to read this report",
        _how_to_read(),
        "## 1. E1 on the holdout global sample",
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
            f"The holdout's seasons follow the same months ({seasons}): summer is December "
            "2025 to February 2026 and winter June to August 2026.",
            _question5_beside(res, cfg),
            "The holdout tables in full:",
            _question5_section(res.hold, cfg),
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


def write_figures(res: HoldoutResults, cfg: AppConfig, fig_dir: Path, passed: bool) -> None:
    ac = cfg.explain_report.associations
    if passed:
        plot_families(res.hold.shap, fig_dir / FIGURES["families"])
        plot_weather(
            res.hold.shap, fig_dir / FIGURES["weather"], cfg.explain_report.shap.weather_range
        )
    for rate, sign, color in (("high", ">", ORANGE), ("low", "<", BLUE)):
        price = ac.high_price if rate == "high" else ac.low_price
        plot_rates(
            res.hold.assoc,
            rate,
            fig_dir / FIGURES[rate],
            f"P(price {sign} ${price:,.0f}/MWh) by same-interval observable "
            f"(holdout, {ac.ci_level:.0%} day-block intervals)",
            color,
            ac.few_days,
        )


def build(
    engine: Engine, cfg: AppConfig, region_id: str, fig_dir: Path, stand_in: bool = False
) -> str:
    res = compute(engine, cfg, region_id, stand_in)
    figs = {k: f"figures/{v}" for k, v in FIGURES.items()}
    text, passed = render(res, cfg, figs)
    found = banned_in(text, cfg.anomalies.evidence.banned_phrases)
    if found:
        raise BannedPhraseError(f"the report holds banned phrases: {found}")
    write_figures(res, cfg, fig_dir, passed)
    return text


def write_report(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    reports = cfg.resolve(cfg.paths.reports_dir)
    text = build(engine, cfg, region_id, reports / "figures")
    path = reports / REPORT_NAME
    path.write_text(text, encoding="utf-8")
    return path

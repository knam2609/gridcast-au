"""Generate ``reports/m3_holdout.md`` (+ figures) from the stored holdout interval run (M3 step 6c).

The contents were fixed before anything was scored (m3_plan.md step 6, question round of
2026-11-13), with M2's holdout report as the model:

* **U5** from what is stored: one holdout interval batch for the frozen recipe, from one run
  whose continuous run reproduced every development interval batch exactly.
* **U1–U4 recomputed on the holdout** with the development report's code, reported as *holds*
  or *does not hold*. They do not change the development verdicts.
* **Development vs holdout**, then **everything step 4 reports**, on the holdout issue times
  (``holdout_main``'s actuals and common mask), with the same tables, figures and bootstrap.
* **A pre-registered winter section:** target months Jun–Aug 2026 (M2's winter under-forecast
  was the known stress test): 80% coverage and the share below and above by lead band, for
  CQR, conformal around LightGBM C, raw QR and the best baseline, with day-block intervals,
  and the settings selected at the retrain whose quarter holds June.
* **The AEMO holdout part:** step 5's comparison on Oct 2025 → Sep 2026, with the pooled
  differences for development and holdout side by side.

Nothing is typed by hand. Every pooled number recomputed from the stored bounds must equal
the stored holdout metric, as in step 4, or the report stops. Analyses added after the
holdout was scored must be labelled as such where they are added.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.evaluation.m2_report import Criterion, LeakageOutcome, criteria_table
from gridcast.evaluation.report import _fmt, _md_table, horizon_ranges
from gridcast.intervals import aemo_report as aemo
from gridcast.intervals.batch import TARGET, interval_config_hash
from gridcast.intervals.evaluation import BOOTSTRAP_DRAWS, U2_RANGE
from gridcast.intervals.holdout import EVALUATION_SET as HOLDOUT_SET
from gridcast.intervals.inputs import read_stored_batch, read_wide
from gridcast.intervals.qr_batch import qr_config_hash
from gridcast.intervals.qr_holdout import EVALUATION_SET as QR_HOLDOUT_SET
from gridcast.intervals.report import (
    BASELINES,
    CONFORMAL_C,
    HEADLINE,
    LABELS,
    LEAKAGE_TESTS,
    METHODS,
    MIN_BOOTSTRAP_DAYS,
    MIN_MONTH_DAYS,
    RAW_QR,
    REARRANGED,
    SHORT,
    WEEK_DAYS,
    IntervalRun,
    Pairs,
    Results,
    _band_text,
    _criteria,
    _interval_runs,
    _nem_date,
    ci,
    ci_signed,
    conditional_methods,
    crossing_facts,
    envelope_facts,
    find_runs,
    group_summary,
    group_table,
    headline_sensitivity,
    hour_table,
    lead_group_tables,
    level_table,
    load_pairs,
    month_name,
    month_note,
    month_table,
    nesting_table,
    other_differences,
    pct,
    plot_coverage_by_lead,
    plot_coverage_by_month,
    plot_is_vs_envelope,
    plot_reliability,
    pooled_table,
    quarter_table,
    reliability_tables,
    run_leakage_tests,
    score_all,
    selection_counts,
    selection_grid,
    signed,
    year_table,
)
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

WINTER: tuple[tuple[int, int], ...] = ((2026, 6), (2026, 7), (2026, 8))  # pre-registered
FIGURES = {
    "coverage": "m3_holdout_coverage_by_lead.png",
    "is80": "m3_holdout_is80_vs_baselines.png",
    "reliability": "m3_holdout_reliability.png",
    "month": "m3_holdout_coverage_by_month.png",
    "aemo_cov": "m3_holdout_aemo_coverage_by_lead.png",
    "aemo_is": "m3_holdout_aemo_is80_by_lead.png",
}


# --------------------------------------------------------------------------- runs


@dataclass(frozen=True, slots=True)
class HoldoutRuns:
    batch_id: str  # the holdout interval batch scored here
    stored_batches: list[str]  # every holdout interval batch of the frozen recipe (U5)
    five: dict[str, IntervalRun]  # 5-minute runs by method (step 4's names)
    thirty: dict[str, IntervalRun]  # half-hour runs by method (step 5's names)
    qr_holdout: IntervalRun  # step 6a's run: fit records and reproduction

    @property
    def config(self) -> dict[str, Any]:
        return self.five[HEADLINE].config


def find_holdout_runs(engine: Engine, cfg: AppConfig, region_id: str) -> HoldoutRuns:
    """The stored holdout interval runs of the frozen recipe, checked against the freeze."""
    iv = cfg.intervals
    if not (iv.frozen_config_hash and iv.frozen_qr_config_hash and iv.frozen_commit):
        raise LookupError("the holdout report runs only after the freeze (intervals.frozen_*)")
    if interval_config_hash(cfg) != iv.frozen_config_hash or (
        qr_config_hash(cfg) != iv.frozen_qr_config_hash
    ):
        raise ValueError("the current interval or QR recipe differs from the frozen one")
    runs = [
        r
        for r in _interval_runs(engine, region_id, HOLDOUT_SET)
        if r.config.get("interval_config_hash") == iv.frozen_config_hash
    ]
    if not runs:
        raise LookupError(
            "no stored holdout interval run for the frozen recipe: run "
            "gridcast evaluate-holdout-intervals first"
        )
    batches = sorted({r.batch_id for r in runs}, key=lambda b: min(
        r.run_id for r in runs if r.batch_id == b
    ))  # fmt: skip
    batch = batches[0]  # the first look; U5 fails if there is more than one
    five = {r.name: r for r in runs if r.batch_id == batch and r.config["resolution_minutes"] == 5}
    thirty = {
        r.name: r for r in runs if r.batch_id == batch and r.config["resolution_minutes"] == 30
    }
    missing = [m for m in (HEADLINE, CONFORMAL_C, RAW_QR, *BASELINES) if m not in five]
    missing += [m for m in aemo.OURS if m not in thirty]
    if missing:
        raise LookupError(f"holdout batch {batch} lacks runs {missing}")
    for run in (*five.values(), *thirty.values()):
        c = run.config
        if c.get("qr_config_hash") != iv.frozen_qr_config_hash:
            raise ValueError(f"{run.name} (run {run.run_id}) has another QR config hash")
        if c.get("frozen_commit") != iv.frozen_commit:
            raise ValueError(f"{run.name} (run {run.run_id}) was built under another freeze")
        if c.get("source_batch") != iv.source_batch:
            raise ValueError(f"{run.name} (run {run.run_id}) was built on another source batch")
    qr_batch = five[RAW_QR].config["qr_holdout_batch"]
    qr_runs = [
        r for r in _interval_runs(engine, region_id, QR_HOLDOUT_SET) if r.batch_id == qr_batch
    ]
    if len(qr_runs) != 1:
        raise LookupError(f"step 6a's QR holdout run {qr_batch} is not stored")
    return HoldoutRuns(batch, batches, five, thirty, qr_runs[0])


def holdout_selections(run: IntervalRun) -> IntervalRun:
    """The run with only the selections that apply to holdout issue times."""
    if "selections" not in run.config:  # raw QR: no conformal selection
        return run
    keep = {pd.Timestamp(r) for r in run.config.get("holdout_retrains", [])}
    config = dict(run.config)
    config["selections"] = [s for s in config["selections"] if pd.Timestamp(s["retrain"]) in keep]
    return dataclasses.replace(run, config=config)


def holdout_retrains(hr: HoldoutRuns) -> list[pd.Timestamp]:
    return [pd.Timestamp(r) for r in hr.config["holdout_retrains"]]


# --------------------------------------------------------------------------- criteria


def u5_criterion(hr: HoldoutRuns, frozen: str) -> Criterion:
    runs = [*hr.five.values(), *hr.thirty.values()]
    shas = sorted({str(r.config.get("code_sha")) for r in runs})
    reproduced = all(bool(r.config.get("reproduction", {}).get("passed")) for r in runs)
    n_dev = len(hr.config["reproduction"]["development_runs"])
    sha_text = ", ".join(f"`{s[:7]}`" for s in shas)
    one = len(hr.stored_batches) == 1 and len(shas) == 1
    return Criterion(
        "U5",
        "the holdout is scored by a single `gridcast evaluate-holdout-intervals` run against "
        "the frozen interval recipe; the command refuses to run again for the same recipe hash",
        f"{len(hr.stored_batches)} holdout interval batch(es) stored for recipe `{frozen}` "
        f"({len(runs)} runs), from run commit(s) {sha_text}; before scoring, the continuous run "
        f"{'reproduced' if reproduced else 'did NOT reproduce'} the {n_dev} stored development "
        "interval runs exactly; `evaluate-holdout-intervals` refuses while this batch exists",
        bool(one and reproduced),
    )


def replication_table(dev: Sequence[Criterion], hold: Sequence[Criterion]) -> str:
    rows = [
        [
            d.key,
            d.rule,
            d.measured,
            "PASS" if d.passed else "FAIL",
            h.measured,
            "**holds**" if h.passed else "**does not hold**",
        ]
        for d, h in zip(dev, hold, strict=True)
    ]
    headers = ["#", "Pass if", "Development", "Dev", "Holdout", "Holdout"]
    return _md_table(headers, rows, "llllll")


# --------------------------------------------------------------------------- winter


def add_winter_groups(
    pairs: Pairs, bands: Sequence[Sequence[int]], winter: Sequence[tuple[int, int]]
) -> list[str]:
    """Groups of pairs whose target falls in the ``winter`` months, per lead band and pooled."""
    labels, _ = pairs.labels["month"]
    codes = [k for k, ym in enumerate(pairs.months) if tuple(ym) in set(winter)]
    in_winter = np.isin(labels, codes)
    n_leads = pairs.actual.shape[1]
    h = np.arange(1, n_leads + 1)
    keys = []
    for lo, hi in [*(tuple(b) for b in bands), (1, n_leads)]:
        key = f"winter_h{int(lo):03d}-{int(hi):03d}"
        pairs.groups[key] = in_winter & ((h >= lo) & (h <= hi))[None, :]
        pairs.group_text[key] = (
            "All leads" if (lo, hi) == (1, n_leads) else _band_text(f"h{lo:03d}-{hi:03d}")
        )
        keys.append(key)
    return keys


def winter_tables(res: Results, keys: Sequence[str]) -> tuple[str, str]:
    methods = conditional_methods(res)
    cov_rows, miss_rows = [], []
    for g in keys:
        e = res.group(HEADLINE, "cov_80", g)
        label = res.pairs.group_text[g]
        if e.n_pairs == 0:
            cov_rows.append([label, "0", "0", *["–"] * len(methods)])
            miss_rows.append([label, *["–"] * len(methods)])
            continue
        cells = []
        for m in methods:
            em = res.group(m, "cov_80", g)
            cells.append(ci(em, 100) if em.n_blocks >= MIN_BOOTSTRAP_DAYS else pct(em.value))
        cov_rows.append([label, f"{e.n_pairs:,}", str(e.n_blocks), *cells])
        miss_rows.append(
            [label]
            + [
                f"{pct(res.group(m, 'below_80', g).value)} / "
                f"{pct(res.group(m, 'above_80', g).value)}"
                for m in methods
            ]
        )
    cov_headers = [
        "Lead band",
        "Pairs",
        "Issue days",
        *[f"{SHORT[m]} 80% (95% CI)" for m in methods],
    ]
    miss_headers = ["Lead band", *[f"{SHORT[m]} below / above %" for m in methods]]
    return (
        _md_table(cov_headers, cov_rows, "l" + "r" * (len(cov_headers) - 1)),
        _md_table(miss_headers, miss_rows, "l" + "r" * (len(miss_headers) - 1)),
    )


def winter_retrain(hr: HoldoutRuns, winter: Sequence[tuple[int, int]]) -> pd.Timestamp | None:
    """The selection retrain whose quarter holds the first winter month's first day."""
    y, m = winter[0]
    first = pd.Timestamp(dt.date(y, m, 1)).tz_localize(NEM_TZ)
    before = [r for r in holdout_retrains(hr) if r <= first]
    return max(before) if before else None


def winter_selection_table(hr: HoldoutRuns, retrain: pd.Timestamp, level: int = 80) -> str:
    rows = []
    bands: list[str] = []
    for m in (HEADLINE, CONFORMAL_C):
        picks = [
            s
            for s in hr.five[m].config["selections"]
            if int(s["level"]) == level and pd.Timestamp(s["retrain"]) == retrain
        ]
        by = {s["band"]: s["chosen"] for s in picks}
        bands = sorted(by)
        rows.append([SHORT[m], *[f"`{by[b]}`" for b in bands]])
    return _md_table(["Method", *[_band_text(b) for b in bands]], rows)


# --------------------------------------------------------------------------- other sections


def period_table(dev: Results, hold: Results) -> str:
    rows = []
    for m in METHODS:

        def cell(stat: str, as_pct: bool, m: str = m) -> str:
            a, b = dev.pooled(m, stat), hold.pooled(m, stat)
            return f"{pct(a)} → {pct(b)}" if as_pct else f"{_fmt(a)} → {_fmt(b)}"

        rows.append(
            [LABELS[m]]
            + [cell(f"cov_{lv}", True) for lv in hold.levels]
            + [cell("is_80", False), cell("wis", False)]
        )
    headers = [
        "Method",
        *[f"Coverage at {lv}% (%)" for lv in hold.levels],
        "IS₈₀ (MW)",
        "WIS (MW)",
    ]
    return _md_table(headers, rows)


def reproduction_table(hr: HoldoutRuns) -> str:
    rows = [
        [
            f"`{r['run']}`",
            f"`{r['batch_id']}`",
            f"{int(r['issue_times']):,}",
            str(r["selections"]),
            "exact" if r["bounds_equal"] and r["selections_equal"] else "DIFFERS",
        ]
        for r in hr.config["reproduction"]["development_runs"]
    ]
    headers = ["Run", "Development batch", "Issue times compared", "Selections", "Result"]
    return _md_table(headers, rows, "llrrl")


def qr_reproduction_text(hr: HoldoutRuns) -> str:
    qc = hr.qr_holdout.config
    q, p = qc["qr_reproduction"], qc["point_reproduction"]
    overlap = int(hr.config["reproduction"]["qr_overlap_issue_times"])
    return (
        f"- **QR (step 6a), before the other retrains were fitted:** the {q['retrain']} "
        f"retrain's {q['fits']} fit records (rows, every candidate's pinball loss and tree "
        f"count, each quantile's selection) and {int(q['pairs']):,} rearranged quantile values "
        f"for {int(q['issue_times']):,} development issue times equal step 2's batch "
        f"`{q['reference_batch']}`: {'exact' if q['passed'] else 'FAILED'}.\n"
        f"- **Point refits (step 6a):** at all {len(qc['retrains'])} retrains, "
        f"{int(p['pairs']):,} point forecasts equal run 2's and `holdout_main`'s stored ones: "
        f"{'exact' if p['passed'] else 'FAILED'}.\n"
        f"- **QR overlap (step 6b):** step 6a's quantiles equal step 2's on {overlap:,} "
        "development issue times."
    )


def _provenance(
    cfg: AppConfig,
    hr: HoldoutRuns,
    res: Results,
    leakage: LeakageOutcome,
    region_id: str,
    report_sha: str | None,
) -> str:
    c = hr.config
    qc = hr.qr_holdout.config
    first = res.pairs.times[0].tz_convert(NEM_TZ)
    last = res.pairs.times[-1].tz_convert(NEM_TZ)
    scored, total = int(res.pairs.mask.sum()), int(res.pairs.mask.size)
    five_ids = ", ".join(
        f"{SHORT[m]} {hr.five[m].run_id}" for m in (HEADLINE, CONFORMAL_C, RAW_QR, *BASELINES)
    )
    retrains = ", ".join(_nem_date(r) for r in c["holdout_retrains"])
    rows = [
        ["Region", region_id],
        ["Holdout interval batch", f"`{hr.batch_id}` (5-minute runs {five_ids}; 5 half-hour runs)"],
        ["Run commit (recorded at the start of the run)", f"`{c['code_sha']}`"],
        [
            "Interval recipe",
            f"`{c['interval_config_hash']}` = `intervals.frozen_config_hash`; QR recipe "
            f"`{c['qr_config_hash']}` = frozen; frozen code at `{c['frozen_commit']}` "
            "(unchanged at the run)",
        ],
        [
            "Point forecasts",
            f"holdout: M2's `holdout_main` `{c['holdout_point_batch']}`; calibration history "
            f"before {first:%Y-%m-%d}: run 2 `{c['source_batch']}`",
        ],
        [
            "Quantile LightGBM C",
            f"step 2 `{c['qr_batch']}` before the holdout, step 6a `{c['qr_holdout_batch']}` "
            f"(run {hr.qr_holdout.run_id}, commit `{(hr.qr_holdout.git_sha or '')[:7]}`, "
            f"{float(qc['seconds']) / 3600:.1f} h) from it",
        ],
        [
            "Continuous run",
            f"issue times from {_nem_date(c['calibration_first_issue'])}; intervals output from "
            f"{_nem_date(c['retrains'][0])}; {float(c['seconds']) / 60:.1f} min",
        ],
        [
            "Holdout issue times (NEM time)",
            f"{first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M}, {len(res.pairs.times):,} "
            "(every 30 min)",
        ],
        [
            "Scored (issue time, lead) pairs, every method",
            f"{scored:,} of {total:,} (`holdout_main`'s common mask)",
        ],
        ["Selection retrains applying to the holdout", retrains],
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


def _aemo_section(
    dev: aemo.Results, hold: aemo.Results, figures: Path
) -> tuple[list[str], aemo.Estimate]:
    title = "VIC1 demand, 30-minute periods, same pairs (holdout, Oct 2025 → Sep 2026)"
    aemo._plot(
        hold,
        "cov",
        figures / FIGURES["aemo_cov"],
        "Share of actuals inside (%)",
        f"80%: coverage. {title}",
    )
    aemo._plot(
        hold, "is", figures / FIGURES["aemo_is"], "Mean interval score (MW)", f"IS80. {title}"
    )
    headers = [
        "Period", "Statistic", "A", "B", "A − B", "95% interval (day blocks)",
        "95% interval (7-day blocks)", "Pairs", "Days",
    ]  # fmt: skip
    rows = [["Development", *r] for r in aemo.difference_rows(dev)]
    rows += [["**Holdout**", *r] for r in aemo.difference_rows(hold)]
    d = hold.difference(aemo.HEADLINE, aemo.PM, "is")
    c = hold.c
    first = c.issue_times[0].tz_convert(NEM_TZ)
    last = c.issue_times[-1].tz_convert(NEM_TZ)
    out = [
        "## AEMO's POE band, holdout part\n",
        "Step 5's comparison (`reports/m3_aemo.md`) repeated on the holdout issue times "
        f"({first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M} NEM time, "
        f"{len(c.issue_times):,} issue times; {c.poe_runs:,} of {c.expected_runs:,} PDPASA `LOR` "
        "runs in the window). Same alignments (matched is the headline), the same pairs rule, "
        "the frozen half-hour recipe, and the same bootstrap. AEMO's band is a benchmark only, "
        f"not pass/fail. {len(c.checked)} pooled 80% statistics of the holdout half-hour runs "
        "were recomputed from the stored bounds and equal the stored metrics.\n",
        aemo._summary(hold) + "\n",
        f"![80% coverage by lead time, holdout](figures/{FIGURES['aemo_cov']})\n",
        "### Share of actual demand inside the 80% interval (%), holdout\n",
        aemo.coverage_table(hold) + "\n",
        "### Misses below and above (%), holdout\n",
        aemo.miss_table(hold) + "\n",
        f"![IS80 by lead time, holdout](figures/{FIGURES['aemo_is']})\n",
        "### Interval score IS80 (MW), holdout\n",
        aemo.is_table(hold) + "\n",
        "### Pooled differences, development and holdout\n",
        _md_table(headers, rows) + "\n",
        "### Sensitivity: the re-centred band and the strict alignment (holdout)\n",
        aemo.sensitivity_table(hold) + "\n",
        "### By quarter of the issue time (holdout)\n",
        aemo.quarter_table(hold) + "\n",
        "### What AEMO's band is (holdout)\n",
        aemo._offset_text(c) + "\n",
    ]
    return out, d


# --------------------------------------------------------------------------- report


def build_report(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    leakage: LeakageOutcome | None = None,
    winter: Sequence[tuple[int, int]] = WINTER,
) -> str:
    report_sha = git_sha()  # before any figure is written, which would make the tree dirty
    hr = find_holdout_runs(engine, cfg, region_id)
    dev_runs = find_runs(engine, cfg, region_id)
    leakage = leakage or run_leakage_tests()
    res_dev = score_all(engine, cfg, region_id, dev_runs)

    c = hr.config
    start, end = pd.Timestamp(c["holdout_start"]), pd.Timestamp(c["holdout_end"])
    levels = sorted(cfg.intervals.levels)
    times, _ = read_wide(
        cfg.data_root() / hr.five[HEADLINE].directory / f"{HEADLINE}__lo{levels[0]}.parquet"
    )
    source = read_stored_batch(
        cfg.data_path(cfg.paths.backtest_dir), str(c["holdout_point_batch"]), TARGET
    ).restrict(start, end)
    pairs = load_pairs(engine, cfg, region_id, times, source=source)
    winter_keys = add_winter_groups(pairs, cfg.ml.lead_bands, winter)
    res = score_all(engine, cfg, region_id, hr.five, split="holdout", pairs=pairs)

    figures = cfg.resolve(cfg.paths.reports_dir) / "figures"
    plot_coverage_by_lead(res, figures / FIGURES["coverage"], "holdout, Oct 2025 → Sep 2026")
    plot_is_vs_envelope(res, figures / FIGURES["is80"])
    plot_reliability(res, figures / FIGURES["reliability"])
    left_out = plot_coverage_by_month(res, figures / FIGURES["month"])

    aemo_dev = aemo.score(aemo.load_comparison(engine, cfg, region_id), cfg.ml.seed)
    aemo_hold = aemo.score(
        aemo.load_comparison(engine, cfg, region_id, holdout_runs=hr.thirty), cfg.ml.seed
    )
    aemo_part, aemo_diff = _aemo_section(aemo_dev, aemo_hold, figures)

    frozen = str(cfg.intervals.frozen_config_hash)
    u5 = u5_criterion(hr, frozen)
    dev_criteria = _criteria(res_dev, leakage)
    hold_criteria = _criteria(res, leakage)
    holds = sum(cr.passed for cr in hold_criteria)
    verdicts = ", ".join(
        f"{cr.key} {'holds' if cr.passed else 'does not hold'}" for cr in hold_criteria
    )
    best = res.best_baseline()
    sens_table, diffs = headline_sensitivity(res)
    cov_lead, is_lead = lead_group_tables(res)
    share_table, pin_table = reliability_tables(res)
    sel_runs = {m: holdout_selections(r) for m, r in hr.five.items()}
    cov80 = 100 * res.per_lead(HEADLINE, "cov_80")
    lo2, hi2 = U2_RANGE
    outside = [int(h) + 1 for h in np.flatnonzero((cov80 < lo2) | (cov80 > hi2))]
    month_cov = res.by_label(HEADLINE, "cov_80", "month")
    full = [k for k, d in enumerate(res.pairs.month_days) if d >= MIN_MONTH_DAYS]
    worst_m = min(full, key=lambda k: month_cov[k]) if full else None
    best_m = max(full, key=lambda k: month_cov[k]) if full else None
    cqr_cov = res.estimate(HEADLINE, "cov_80")
    cqr_cov_w = res.estimate(HEADLINE, "cov_80", res.weeks)
    d_best = next(d for o, n, d, _ in diffs if o == best and n == "IS₈₀")
    d_conf = next(d for o, n, d, _ in diffs if o == CONFORMAL_C and n == "IS₈₀")
    w_conf = next(d for o, n, d, _ in diffs if o == CONFORMAL_C and n == "WIS")
    rearr = res.difference(REARRANGED, HEADLINE, "wis")
    is_best = res.pooled(best, "is_80")
    is_conf = res.pooled(CONFORMAL_C, "is_80")
    w_all = res.group(HEADLINE, "cov_80", winter_keys[-1])
    w_cov, w_miss = winter_tables(res, winter_keys)
    w_retrain = winter_retrain(hr, winter)
    winter_text = ", ".join(month_name(*ym) for ym in winter)
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    known_groups = list(res.pairs.known_groups)

    if w_all.n_pairs:
        w_below = res.group(HEADLINE, "below_80", winter_keys[-1]).value
        w_above = res.group(HEADLINE, "above_80", winter_keys[-1]).value
        winter_summary = (
            f"- **Winter ({winter_text}, pre-registered):** CQR covers {pct(w_all.value)}% at "
            f"80% (95% CI {pct(w_all.lo)}–{pct(w_all.hi)}%), missing below on {pct(w_below)}% "
            f"and above on {pct(w_above)}% of pairs. Conformal around LightGBM C covers "
            f"{pct(res.group(CONFORMAL_C, 'cov_80', winter_keys[-1]).value)}%."
        )
    else:
        winter_summary = f"- **Winter ({winter_text}):** no scored holdout pair targets it."
    month_summary = (
        f" **By month** it ranges from {pct(month_cov[worst_m])}% "
        f"({month_name(*res.pairs.months[worst_m])}) to {pct(month_cov[best_m])}% "
        f"({month_name(*res.pairs.months[best_m])})."
        if worst_m is not None and best_m is not None
        else ""
    )

    out = [
        "# Milestone 3: prediction intervals for Victorian demand, the holdout scored once\n",
        f"> **Generated** by `gridcast report-m3-holdout` on {generated} from the stored holdout "
        "interval run (issue times "
        f"{res.pairs.times[0].tz_convert(NEM_TZ):%Y-%m-%d} → "
        f"{res.pairs.times[-1].tz_convert(NEM_TZ):%Y-%m-%d}, NEM time). Do not edit by hand. "
        "The interval recipe was frozen before the AEMO comparison and before any holdout "
        "interval existed, and the report's contents were fixed before the holdout was scored "
        "(m3_plan.md step 6).\n",
        "## Summary\n",
        (
            "- **U5 passes:** the holdout intervals were scored by one "
            "`evaluate-holdout-intervals` run of the frozen recipe, after its continuous run "
            "reproduced every stored development interval exactly."
            if u5.passed
            else f"- **U5 FAILS:** {u5.measured}."
        ),
        f"- **{holds} of {len(hold_criteria)} development criteria hold on the holdout** "
        f"({verdicts}; replication, reported, not pass/fail).",
        f"- **Calibration:** CQR's 80% intervals cover {pct(cqr_cov.value, 2)}% of actual "
        f"demand on the holdout (95% CI {pct(cqr_cov.lo, 2)}–{pct(cqr_cov.hi, 2)}% with day "
        f"blocks, {pct(cqr_cov_w.lo, 2)}–{pct(cqr_cov_w.hi, 2)}% with {WEEK_DAYS}-day blocks), "
        f"against {pct(res_dev.pooled(HEADLINE, 'cov_80'), 2)}% in development; "
        f"{pct(res.pooled(HEADLINE, 'cov_50'))}% at 50% and {pct(res.pooled(HEADLINE, 'cov_95'))}% "
        f"at 95%. The raw quantile LightGBM C bounds cover {pct(res.pooled(RAW_QR, 'cov_80'))}% "
        "at 80%.",
        f"- **Sharpness:** CQR's pooled IS₈₀ is {_fmt(res.pooled(HEADLINE, 'is_80'))} MW, "
        f"against {_fmt(is_best)} MW for the best baseline interval ({LABELS[best].lower()}; "
        f"{signed(100 * d_best.value / is_best)}%) and {_fmt(is_conf)} MW for conformal "
        f"intervals around the same point forecast (difference {ci_signed(d_conf)} MW). Over "
        f"all three levels, the WIS difference to conformal C is {ci_signed(w_conf)} MW"
        + ("; its 95% interval includes zero." if w_conf.lo < 0 < w_conf.hi else "."),
        f"- **By lead time:** 80% coverage stays within {lo2:g}–{hi2:g}% at "
        f"{len(cov80) - len(outside)} of {len(cov80)} leads"
        + (f" (outside at {horizon_ranges(outside)})" if outside else "")
        + f", from {_fmt(float(np.nanmin(cov80)), 2)}% to {_fmt(float(np.nanmax(cov80)), 2)}%."
        + month_summary,
        winter_summary,
        f"- **Against AEMO's POE band (holdout part):** AEMO's band (matched) contains "
        f"{pct(aemo_hold.pooled(aemo.PM, 'cov'))}% of actual 30-minute demand and CQR's 80% "
        f"interval {pct(aemo_hold.pooled(aemo.HEADLINE, 'cov'))}%; CQR's IS80 is "
        f"{aemo.signed(aemo_diff.value)} MW against AEMO's (95% CI {aemo.ci_text(aemo_diff)} "
        "MW).\n",
        "## Provenance\n",
        _provenance(cfg, hr, res, leakage, region_id, report_sha) + "\n",
        "## U5: holdout used once\n",
        criteria_table([u5]) + "\n",
        "**Reproduction before scoring (step 6 decision).** The locked run built every interval "
        "over one continuous run of issue times from "
        f"{_nem_date(c['calibration_first_issue'])}, so calibration windows, ACI state and "
        "selections carry across the holdout boundary. Before any holdout pair was scored, "
        "the development part of each result had to equal the stored development batch "
        "exactly, bounds and selections:\n",
        reproduction_table(hr) + "\n",
        qr_reproduction_text(hr) + "\n",
        "## Replication of U1–U4 on the holdout\n",
        "The development criteria, recomputed on the holdout issue times with the same code. "
        "They are reported as *holds* or *does not hold* and do not change the development "
        "verdicts (m3_plan.md §3). U4 is the same leakage test suite, run by this command.\n",
        replication_table(dev_criteria, hold_criteria) + "\n",
        f"\n**{holds} of {len(hold_criteria)} hold on the holdout.**\n",
        "## Development vs holdout\n",
        "Pooled over every lead time, on each period's own scored pairs (development: run 2's "
        f"common mask from {cfg.intervals.first_issue}; holdout: `holdout_main`'s). Each cell "
        "reads development → holdout.\n",
        period_table(res_dev, res) + "\n",
        f"## Winter, {winter_text} (pre-registered)\n",
        "M2's holdout found its largest error in the winter: LightGBM C under-forecast demand, "
        "worst in Jun 2026 (`reports/m2_holdout.md`, *Bias by month*). This section was fixed "
        "before the holdout intervals were scored. It shows how the intervals cope with a "
        "point forecast that is biased for weeks: pairs whose target falls in these months, "
        "by M2's lead bands.\n",
        "80% coverage (%), with 95% intervals over whole issue days where the pairs come from "
        f"at least {MIN_BOOTSTRAP_DAYS} issue days:\n",
        w_cov + "\n",
        "Share of actuals below and above the 80% interval (%):\n",
        w_miss + "\n",
        (
            f"Settings selected at the {w_retrain.tz_convert(NEM_TZ):%Y-%m-%d} retrain, whose "
            "quarter holds the winter's first month (80% level; `w<weeks>_<grouping>_g<γ>`, "
            "where γ is the ACI step):\n\n" + winter_selection_table(hr, w_retrain) + "\n"
            if w_retrain is not None
            else "No selection retrain precedes the winter months.\n"
        ),
        "## Methods\n",
        "As in the development report (`reports/m3_intervals.md`, *Methods*): CQR on quantile "
        "LightGBM C (the headline), conformal intervals around LightGBM C and around the three "
        "baselines, raw QR, and CQR rearranged (reported only). Settings are re-selected at "
        "every quarterly retrain from past data only, with the frozen recipe.\n",
        "## Pooled results (holdout)\n",
        pooled_table(res) + "\n",
        "Per level, with 95% intervals for coverage (block bootstrap over whole issue days) "
        "and the share of actuals below and above the interval:\n",
        level_table(res) + "\n",
        "### Headline differences and the block-length sensitivity (holdout)\n",
        "Negative differences mean CQR scores better.\n",
        sens_table + "\n",
        "Other differences (reported only):\n",
        other_differences(res) + "\n",
        "## By lead time (holdout)\n",
        f"![80% coverage by lead, holdout](figures/{FIGURES['coverage']})\n",
        f"80% coverage by lead time, with U2's {lo2:g}–{hi2:g}% range shaded.\n",
        "80% coverage (%) by lead-time group:\n",
        cov_lead + "\n",
        f"![IS80 against the baseline envelope, holdout](figures/{FIGURES['is80']})\n",
        "The dashed line is the lowest baseline IS₈₀ at each lead time ("
        + "; ".join(envelope_facts(res))
        + ").\n",
        "IS₈₀ (MW) by lead-time group:\n",
        is_lead + "\n",
        "## Reliability and quantile loss (holdout)\n",
        f"![Reliability diagram, holdout](figures/{FIGURES['reliability']})\n",
        "Share at or below each bound (%):\n",
        share_table + "\n",
        "Mean pinball loss per quantile level (MW):\n",
        pin_table + "\n",
        "## Nesting of the levels (holdout, disclosure)\n",
        nesting_table(res) + "\n",
        f"Sorting CQR's bounds per pair changes its WIS by {ci_signed(rearr, 2)} MW and its "
        f"80% coverage from {pct(res.pooled(HEADLINE, 'cov_80'), 2)}% to "
        f"{pct(res.pooled(REARRANGED, 'cov_80'), 2)}%.\n",
        "## Conditional coverage (holdout)\n",
        f"Shown for CQR, conformal around LightGBM C, raw QR and the best baseline interval "
        f"on the holdout ({LABELS[best].lower()}, lowest pooled IS₈₀).\n",
        f"![80% coverage by month, holdout](figures/{FIGURES['month']})\n",
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
        month_note(res, cfg, cfg.region(region_id).local_tz) + "\n",
        "### By target hour\n",
        hour_table(res) + "\n",
        "### By year\n",
        year_table(res) + "\n",
        "### By retrain quarter\n",
        quarter_table(res, holdout_retrains(hr)) + "\n",
        "## Extreme events (holdout, reported only)\n",
        "Coverage intervals resample whole issue days, shown only for groups whose pairs come "
        f"from at least {MIN_BOOTSTRAP_DAYS} issue days.\n",
        "### Groups known at the issue time\n",
        group_table(res, known_groups) + "\n",
        "Summary: " + "; ".join(group_summary(res, g) for g in known_groups) + ".\n",
        "### Groups defined by the outcome\n",
        "Biased downwards by construction (see the development report). The top-1% threshold "
        "is the 99th percentile of actual demand over the scored holdout target intervals; "
        "named events outside the holdout have no pairs.\n",
        group_table(res, res.pairs.outcome_groups) + "\n",
        "## Per-retrain selections (holdout)\n",
        "Selections that apply to holdout issue times (the first is the last development "
        "retrain, whose quarter runs into the holdout):\n",
        selection_counts(sel_runs, res.levels) + "\n",
        "CQR's 80% selection at each of these retrains and lead bands:\n",
        selection_grid(sel_runs[HEADLINE], 80) + "\n",
        "Crossed CQR intervals in the holdout, output as zero width at the midpoint: "
        + ", ".join(
            f"{int(v):,} at {k}%" for k, v in hr.five[HEADLINE].config["zero_width_holdout"].items()
        )
        + ".\n",
        "### Quantile crossing (raw QR, holdout)\n",
        "\n".join(
            f"- {line}"
            for line in crossing_facts(
                engine,
                hr.five[RAW_QR],
                split="holdout",
                config=hr.qr_holdout.config,
                forecast_from=_nem_date(hr.qr_holdout.config["forecast_first_issue"]),
            )
        )
        + "\n",
        "## Leakage tests (U4)\n",
        "Run by this command: " + ", ".join(f"`{t}`" for t in LEAKAGE_TESTS) + ". "
        f"**{leakage.passed} passed, {leakage.failed} failed.**\n",
        *aemo_part,
        "## Disclosures\n",
        "- **The holdout is new for intervals, not for point forecasts.** Its point errors, "
        "including the winter under-forecast, were seen in M2 before the interval recipe was "
        "designed (m3_plan.md §1). The recipe's settings are re-selected from past data only, "
        "but its design was not blind to them.",
        "- **The winter section was chosen knowing M2's winter bias.** It was fixed before any "
        "holdout interval was scored, so its numbers were not looked for.",
        "- **Two stages.** Step 6a fitted QR over the holdout and stored its quantiles without "
        "computing any metric. This run scored them, under the U5 lock, with everything else.",
        "- **One look.** The holdout intervals are scored once (U5). A problem found later is "
        "disclosed here, not fixed by rescoring.",
        f"- {res.pairs.mask.size - int(res.pairs.mask.sum()):,} holdout pairs without C's "
        "weather inputs are excluded for every method (`holdout_main`'s common mask).",
    ]
    return "\n".join(out) + "\n"


def write_report(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    leakage: LeakageOutcome | None = None,
    winter: Sequence[tuple[int, int]] = WINTER,
) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / "m3_holdout.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id, leakage, winter))
    return path

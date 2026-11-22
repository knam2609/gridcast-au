"""``reports/m4_holdout.md``: the anomaly holdout of both recipe versions (m4_plan.md step 6;
``gridcast report-m4-holdout``).

The readings were fixed before either holdout run (m4_plan.md §1, "Clarified in step 6"):

* every number is recomputed from the stored holdout runs (tiers, lines, events and counts,
  as :func:`~gridcast.anomalies.report.recompute` does), and the report stops on a mismatch;
* A1, A3, A4, A5 and E1 are replications, reported as *holds* or *does not hold* (§3.2),
  with each version's development verdict beside them (v1 pre-registered, v2 post hoc);
  E1 holds if it holds at the holdout demand peaks of the run and on ``explain-holdout``'s
  global sample;
* A2 has no class-P event in the holdout (§3.1); the class-O day is reported;
* A6 is the holdout's own criterion: one stored holdout run per recipe hash, and
  ``evaluate-holdout-anomalies`` refuses another (checked live by this report);
* v2's D1, D2, D3 and D5 must equal v1's on the holdout run too (§2.10);
* the case studies are the 5 holdout events ranked first by the case rule, per version, with
  no named days.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.anomalies.freeze import freeze_state
from gridcast.anomalies.report import (
    A3_POWER,
    NAMES,
    CausalityOutcome,
    Results,
    _case_studies,
    _catalogue,
    _overlap,
    _power_by_year,
    _power_table,
    _reference_table,
    _thresholds_table,
    a1_passed,
    check_unchanged,
    compute,
    criteria,
    pct,
    plot_monthly_rates,
    plot_power,
    run_causality_tests,
    size_label,
)
from gridcast.config import AppConfig
from gridcast.evaluation.m2_report import Criterion
from gridcast.evaluation.report import _md_table
from gridcast.explainability.global_run import read_global
from gridcast.provenance import git_sha

REPORT_NAME = "m4_holdout.md"
FIGURE_PREFIX = {1: "m4h_v1", 2: "m4h_v2"}
VERSIONS = (1, 2)
# The criteria replicated on the holdout (§3.2); E1 counts once, holding only if both parts do.
REPLICATED = ("A1", "A3", "A4", "A5", "E1")
E1_PEAKS = "E1 (demand peaks)"
E1_GLOBAL = "E1 (global sample)"


@dataclass(slots=True)
class VersionResults:
    version: int
    holdout: Results
    dev: Results


@dataclass(frozen=True, slots=True)
class Row:
    """One criterion: its rule, and each version's holdout and development outcome."""

    key: str
    rule: str
    holdout: dict[int, Criterion | None]  # None: not applicable on the holdout (A2)
    dev: dict[int, Criterion | None]  # None: no development counterpart (A6)


# --------------------------------------------------------------------------- criteria


def gate_criterion(gate: dict, what: str) -> Criterion:
    return Criterion(
        E1_GLOBAL,
        f"§2.6's gate on every pair of {what}",
        f"{gate['n']:,} pairs; {gate['forecast_mismatches']} forecast mismatches, "
        f"{gate['sum_failures']} sum failures (largest error {gate['max_sum_error']:.1e})",
        bool(gate["passed"]),
    )


def a6_criterion(res: Results, refused: str | None) -> Criterion:
    """One stored holdout run for the hash (``load_run`` requires exactly one), and the
    command refuses another (``refused``: its refusal, None if it would run)."""
    locked = refused is not None and "(A6)" in refused
    return Criterion(
        "A6",
        "the holdout is scored by a single `evaluate-holdout-anomalies` run per recipe hash, "
        "and the command refuses to run again",
        f"run {res.run.run_id} (batch `{res.run.batch_id[:8]}`), the only holdout run stored "
        f"for hash `{res.run.config['anomaly_config_hash']}`; a second run "
        + ("is refused" if locked else "is **not** refused"),
        locked,
    )


def rows_for(
    versions: Sequence[VersionResults],
    gates: tuple[dict, dict],
    refusals: dict[int, str | None],
) -> list[Row]:
    """The verdict rows: A1–A5 and E1 at the demand peaks from each run, E1 on the global
    samples (development, holdout), and A6."""
    hold = {v.version: {c.key: c for c in criteria(v.holdout)} for v in versions}
    dev = {v.version: {c.key: c for c in criteria(v.dev)} for v in versions}
    keys = list(next(iter(hold.values())))
    rows = []
    for k in keys:
        first = hold[versions[0].version][k]
        rows.append(
            Row(
                k,
                first.rule,
                {v: (None if k == "A2" else hold[v][k]) for v in hold},
                {v: dev[v][k] for v in dev},
            )
        )
    dev_gate, hold_gate = gates
    g_dev = gate_criterion(dev_gate, "the development global sample")
    g_hold = gate_criterion(hold_gate, "the global sample")
    rows.append(
        Row(
            E1_GLOBAL,
            "§2.6's gate on every pair of the global SHAP sample (`explain-dev` / "
            "`explain-holdout`; one sample serves both versions)",
            {v: g_hold for v in hold},
            {v: g_dev for v in dev},
        )
    )
    by_version = {v.version: v for v in versions}
    rows.append(
        Row(
            "A6",
            a6_criterion(versions[0].holdout, refusals.get(versions[0].version)).rule,
            {v: a6_criterion(by_version[v].holdout, refusals.get(v)) for v in hold},
            {v: None for v in dev},
        )
    )
    return rows


def replicated_holds(rows: Sequence[Row], version: int) -> tuple[int, int]:
    """How many of A1, A3, A4, A5 and E1 hold on the holdout for ``version``."""
    got = {r.key: r.holdout[version] for r in rows}
    held = 0
    for k in REPLICATED:
        if k == "E1":
            parts = [got[E1_PEAKS], got[E1_GLOBAL]]
            held += all(c is not None and c.passed for c in parts)
        else:
            c = got[k]
            held += c is not None and c.passed
    return held, len(REPLICATED)


def holdout_cell(c: Criterion | None, key: str) -> str:
    if c is None:
        return "– (no class-P event in the holdout; §3.1)"
    if key == "A6":
        return ("**PASS**" if c.passed else "**FAIL**") + f": {c.measured}"
    return ("**holds**" if c.passed else "**does not hold**") + f": {c.measured}"


def dev_cell(c: Criterion | None) -> str:
    if c is None:
        return "–"
    return ("PASS" if c.passed else "FAIL") + f": {c.measured}"


def verdict_table(rows: Sequence[Row]) -> str:
    body = [
        [
            r.key,
            r.rule,
            holdout_cell(r.holdout[1], r.key),
            dev_cell(r.dev[1]),
            holdout_cell(r.holdout[2], r.key),
            dev_cell(r.dev[2]),
        ]
        for r in rows
    ]
    return _md_table(
        [
            "#",
            "Pass if",
            "v1 holdout",
            "v1 development (pre-registered)",
            "v2 holdout (blind)",
            "v2 development (post hoc)",
        ],
        body,
        "llllll",
    )


# --------------------------------------------------------------------------- tables


def a1_beside(hold: Results, dev: Results) -> str:
    """The holdout's A1 table, with the development shares beside each detector."""
    d = dev.a1.set_index("detector")
    rows = []
    for _, r in hold.a1.iterrows():
        ok1, ok2 = a1_passed(r)
        dr = d.loc[r["detector"]]
        rows.append(
            [
                NAMES[r["detector"]],
                f"{r['scored']:,}",
                pct(r["p01"]),
                "in" if ok1 else "**out**",
                pct(r["p001"], 3),
                "in" if ok2 else "**out**",
                pct(r["p0001"], 4),
                f"{pct(dr['p01'])} / {pct(dr['p001'], 3)}",
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
            "Development: p ≤ 0.01 / p ≤ 0.001",
        ],
        rows,
        "lrrcrcrr",
    )


def a3_beside(hold: Results, dev: Results) -> str:
    got_dev = {(d, s): (n, p) for d, s, n, p in dev.a3}
    rows = []
    for d, s, n, p in hold.a3:
        dn, dp = got_dev.get((d, s), (0, np.nan))
        rows.append(
            [
                NAMES[d],
                size_label(d, s),
                f"{n:,}",
                pct(p, 1),
                "holds" if (not pd.isna(p) and p >= A3_POWER) else "**does not hold**",
                f"{pct(dp, 1)} (n {dn:,})",
            ]
        )
    return _md_table(
        ["Detector", "Largest size", "Injections", "Detected", "≥ 90%", "Development"],
        rows,
        "llrrcr",
    )


def runs_table(versions: Sequence[VersionResults], cfg: AppConfig) -> str:
    def cells(v: VersionResults) -> list[str]:
        res = v.holdout
        c = res.run.config
        rep = c.get("reproduction", {})
        record = cfg.anomalies if v.version == 1 else cfg.anomalies_v2
        state = freeze_state(cfg, version=v.version)
        full = ", ".join(f"{x['name']} {x['items']:,}" for x in rep.get("full", []))
        part = ", ".join(f"{x['name']} {x['items']:,}" for x in rep.get("development_part", []))
        return [
            f"{res.run.run_id} (batch `{res.run.batch_id}`)",
            f"`{c['anomaly_config_hash']}` (frozen `{record.frozen_config_hash}`)",
            f"`{c['frozen_commit']}`; run from `{(res.run.git_sha or '')[:7]}`",
            f"development run {rep.get('development_run', '–')} recomputed exactly "
            f"({rep.get('seconds', 0):,.0f} s): {full}",
            f"equal to the stored development batch: {part}",
            f"{c['series']['intervals']:,} intervals, {c['series']['first_end']} → "
            f"{c['series']['last_end']} (UTC interval ends)",
            f"every detector and the lines from {c['periods']['market_first_day']} to "
            f"{c['periods']['end_day']} (exclusive, NEM days)",
            "QR batches "
            + " + ".join(
                f"`{b[:8]}`"
                for b in c["inputs"].get("qr_batches", [c["inputs"].get("qr_batch", "")])
            )
            + f", split at the holdout start: {c['inputs']['band_issue_times']:,} issue times; "
            f"actuals equal market demand on {c['inputs']['checked_actuals']:,} pairs",
            ", ".join(
                f"{k} {c['seconds'][k]:,.0f} s"
                for k in ("load", "detect", "inject", "cuts", "evidence")
                if k in c["seconds"]
            ),
            "holds" if not state.problems else "; ".join(state.problems),
        ]

    labels = [
        "Run",
        "Anomaly config hash",
        "Frozen recipe code",
        "Reproduction, 1",
        "Reproduction, 2 (the continuous run's development part)",
        "Series",
        "Scored",
        "D2/D3 band",
        "Run time",
        "Freeze now",
    ]
    cols = [cells(v) for v in versions]
    rows = [[lab, *(c[i] for c in cols)] for i, lab in enumerate(labels)]
    rows.append(["Report code", *[f"`{git_sha() or 'unknown'}`"] * len(versions)])
    return _md_table(["", *(f"v{v.version}" for v in versions)], rows, "l" + "l" * len(versions))


# --------------------------------------------------------------------------- the report


def _intro() -> str:
    return (
        "Generated by `gridcast report-m4-holdout` from the two stored holdout runs, one per "
        "recipe version, each scored once (A6). Every number below is recomputed from the "
        "stored files and events, and the report stops if a recomputed tier, line, event or "
        "count differs from a run's record. The criteria were pre-registered in "
        "[`docs/plans/m4_plan.md`](../docs/plans/m4_plan.md) §3, and this report's readings "
        'were fixed before either holdout run (§1, "Clarified in step 6"). On the holdout, '
        "A1, A3, A4, A5 and E1 are replications, reported as *holds* or *does not hold*. They "
        "do not change the development verdicts (§3.2). A6 is the holdout's own criterion.\n\n"
        "- **v1** is the pre-registered recipe, and this is its replication.\n"
        "- **v2** is the disclosed revision of D4 and D6 (§2.10), designed after v1's "
        "development results, and this is its first blind test. M5 and M6 serve v2 whatever "
        "the holdout shows; that was fixed before any holdout score.\n"
        "- The holdout year is NEM days 2025-10-01 → 2026-09-30. It has no class-P reference "
        "event, so A2 is judged on development only (§3.1).\n"
    )


def _disclosures() -> str:
    return (
        "- The detectors were designed after M2's and M3's development and holdout results "
        "had been seen. The planning saw some holdout facts (plan §3.3): the price calm "
        "(0.2% of intervals above $300, 25.9% negative), one day above $5,000 (8 Jul 2026), "
        "the demand jumps of 6 and 10 Nov 2025, and the flagged day of 23 Mar 2026.\n"
        "- v2 was designed after v1's development verdicts, so its development verdicts are "
        "post hoc. Its holdout run is blind: no holdout score of either version existed when "
        "v2 was frozen.\n"
        '- The order was fixed before the runs (plan §1, "Clarified in step 6"). '
        "`explain-holdout` ran first and printed only its E1 gates and timings. This report's "
        "code was committed before either anomaly holdout run, after a test on the stored "
        "development runs. v1 and v2 then ran back to back, and their console output went to "
        "log files that were not read until both runs were stored.\n"
        "- Before the runs, `evaluate-holdout-anomalies --check-only` ran for both versions. "
        "It prints input counts only (series length, band issue times, the actual checks, the "
        "number of holdout models), and no score.\n"
    )


def render(
    versions: Sequence[VersionResults],
    cfg: AppConfig,
    tz: str,
    gates: tuple[dict, dict],
    refusals: dict[int, str | None],
    unchanged: int,
    explain_runs: int,
    figures: dict[int, dict[str, str]],
) -> str:
    rows = rows_for(versions, gates, refusals)
    held = {v.version: replicated_holds(rows, v.version) for v in versions}
    a6_row = next(r for r in rows if r.key == "A6")
    a6 = {
        v.version: bool(c is not None and c.passed)
        for v in versions
        for c in [a6_row.holdout[v.version]]
    }
    summary = "; ".join(
        f"**v{v}: {h} of {n} replicated criteria hold** (A1, A3, A4, A5, E1), A6 "
        + ("passes" if a6[v] else "**fails**")
        for v, (h, n) in held.items()
    )
    out = [
        "# M4: market anomalies on the holdout year (Victoria), v1 and v2\n",
        _intro(),
        "## Verdicts (holdout, with development beside)\n",
        verdict_table(rows) + "\n",
        summary + ". The development verdicts stand as published.\n",
        "## Runs\n",
        runs_table(versions, cfg) + "\n",
        f"v2's D1, D2, D3 and D5 are v1's: {unchanged} stored columns of the two holdout runs "
        "(scores, p-values, window sizes, tiers and scored-period flags, every interval of the "
        "continuous run) are equal (§2.10).\n",
        "## A1: alert budget\n",
        "Each detector's p-value is a conformal rank against its own trailing 26 weeks, so "
        "under exchangeability p ≤ 0.001 should flag about 0.1% of scored intervals. A1 allows "
        "a factor of 2 either way. *Scored* means inside the holdout with a p-value (D2/D3: "
        "pairs in M2's common mask; D6: a month model). The trailing windows cross into the "
        "holdout from development, as a deployed detector's would.\n",
    ]
    for v in versions:
        out += [
            f"**v{v.version}:**\n",
            a1_beside(v.holdout, v.dev) + "\n",
            f"![v{v.version}: monthly alert rates]({figures[v.version]['monthly']})\n",
        ]
    out += [
        "## The class-O day (reported)\n",
        "The holdout's one reference event is class O (it may leave no visible trace), so it "
        "is reported, not judged. An alert counts if a calibrated detector raised a notable "
        "alert at an interval ending within 30 min of the flagged span.\n",
    ]
    for v in versions:
        o = v.holdout.refs_o
        out += [
            f"**v{v.version}:** {sum(c.caught for c in o)} of {len(o)} holdout class-O events "
            "had a calibrated alert in that window.\n",
            _reference_table(o, tz) + "\n",
        ]
    out += [
        "## A3: detection power\n",
        "As in development: each injection is scored alone against the unmodified profiles, "
        "baselines, bands, month models and windows, at starts drawn from the holdout's scored "
        "intervals where the unmodified detector raised no notable alert within the span ± 1 h. "
        "Detected = at least one notable alert from that detector inside the span.\n",
    ]
    for v in versions:
        res = v.holdout
        flat = res.power[res.power["reported_only"]]
        flat_line = (
            f"{pct(float(flat['power'].iloc[0]), 1)} of {int(flat['n'].iloc[0]):,}"
            if len(flat)
            else "not run"
        )
        out += [
            f"### v{v.version}\n",
            a3_beside(res, v.dev) + "\n",
            "All sizes (reported):\n",
            _power_table(res) + "\n",
            "Smallest size reaching 50% and 90% (reported):\n",
            _thresholds_table(res) + "\n",
            "Power at the largest size by start year (reported):\n",
            _power_by_year(res) + "\n",
            f"Reported only: holding the price flat for 3 h was detected in {flat_line}.\n",
            f"![v{v.version}: power by size]({figures[v.version]['power']})\n",
        ]
    out += ["## A4: evidence without causal claims\n"]
    for v in versions:
        ev = v.holdout.run.events
        out.append(
            f"- v{v.version}: {len(ev):,} holdout events; {v.holdout.a4[0]} incomplete, "
            f"{v.holdout.a4[1]} with a banned phrase.\n"
        )
    out += ["## A5: causality\n"]
    for v in versions:
        res = v.holdout
        cuts = ", ".join(str(t) for t in pd.to_datetime(res.run.cuts["cut_end"], utc=True))
        out.append(
            f"- v{v.version}: the synthetic suite's v{v.version} cases (run by this report) "
            f"{res.causality.passed} passed, {res.causality.failed} failed, a leaky control "
            f"included; the run repeated detection truncated at {len(res.run.cuts)} seeded "
            f"cuts inside the holdout ({cuts}), with {res.cut_differences} differences at or "
            f"before them; {len(res.run.forests)} monthly models, {len(res.forest_leaks)} of "
            "which saw an interval ending after their month's start.\n"
        )
    gd, gh = gates
    out += [
        "## E1: SHAP reproduces the forecasts\n",
        "- At the holdout demand peaks of each run: "
        + "; ".join(
            f"v{v.version} {v.holdout.run.config['shap_gate']['n']:,} pairs, "
            f"{v.holdout.run.config['shap_gate']['forecast_mismatches']} mismatches"
            for v in versions
        )
        + ".\n",
        f"- On the holdout global sample (`explain-holdout`, {explain_runs} run"
        f"{'s' if explain_runs != 1 else ''}): {gh['n']:,} pairs, "
        f"{gh['forecast_mismatches']} forecast mismatches, {gh['sum_failures']} SHAP-sum "
        f"failures (largest error {gh['max_sum_error']:.1e}). Its summaries are in "
        "[`m4_explainability_holdout.md`](m4_explainability_holdout.md). Development: "
        f"{gd['n']:,} pairs, {gd['forecast_mismatches']} mismatches.\n",
        "## Event catalogue (reported)\n",
    ]
    for v in versions:
        res = v.holdout
        catalogue, categories = _catalogue(res)
        lines_only = res.run.events[res.run.events["lines_only"]]
        out += [
            f"### v{v.version}\n",
            catalogue + "\n",
            categories + "\n",
            f"Price-line-only events: {len(lines_only):,}.\n",
            "Calibrated events in which both detectors fired (diagonal: events with that "
            "detector):\n",
            _overlap(res) + "\n",
        ]
    out += [
        "## Case studies\n",
        f"Chosen by the fixed rule (§3.2 and the step-4 tie rule): the {cfg.anomalies.cases.n} "
        "holdout events with the lowest p from a calibrated detector, ties by more calibrated "
        "detectors firing, then a seeded draw. The holdout has no named days. Summaries are "
        "generated from the evidence by fixed templates and state facts only.\n",
    ]
    for v in versions:
        out += [f"### v{v.version}\n", _case_studies(v.holdout, cfg, tz, named=False) + "\n"]
    out += ["## Disclosures\n", _disclosures()]
    return "\n".join(out)


@dataclass(slots=True)
class HoldoutReport:
    text: str
    versions: list[VersionResults]


def compute_versions(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    split: str = "holdout",
    causality: dict[int, CausalityOutcome] | None = None,
) -> list[VersionResults]:
    """Both versions' results on ``split`` and on development (``split="dev"`` lets a test
    render the report from the development runs alone)."""
    out = []
    for version in VERSIONS:
        suite = (causality or {}).get(version) or run_causality_tests(version)
        hold = compute(engine, cfg, region_id, suite, version, split)
        dev = hold if split == "dev" else compute(engine, cfg, region_id, suite, version, "dev")
        out.append(VersionResults(version, hold, dev))
    return out


def live_refusals(engine: Engine, cfg: AppConfig, region_id: str) -> dict[int, str | None]:
    """What ``evaluate-holdout-anomalies`` says now, per version (None: it would run)."""
    from gridcast.anomalies.holdout import HoldoutRefused, preflight

    out: dict[int, str | None] = {}
    for version in VERSIONS:
        try:
            preflight(engine, cfg, region_id, version)
            out[version] = None
        except HoldoutRefused as exc:
            out[version] = str(exc)
    return out


def build_report(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    figures_dir: Path,
    split: str = "holdout",
    causality: dict[int, CausalityOutcome] | None = None,
) -> HoldoutReport:
    versions = compute_versions(engine, cfg, region_id, split, causality)
    unchanged = check_unchanged(versions[0].holdout.run, versions[1].holdout.run)
    dev_sample = read_global(cfg, "dev")
    hold_sample = read_global(cfg, "holdout")
    gates = (dev_sample.manifest["gate"], hold_sample.manifest["gate"])
    explain_runs = 1 + len(hold_sample.manifest.get("previous_runs", []))
    figures: dict[int, dict[str, str]] = {}
    for v in versions:
        prefix = FIGURE_PREFIX[v.version]
        res = v.holdout
        plot_power(res.power, figures_dir / f"{prefix}_power_by_size.png", res.rc.detectors)
        plot_monthly_rates(
            res.monthly, figures_dir / f"{prefix}_alert_rate_by_month.png", res.rc.detectors
        )
        figures[v.version] = {
            "power": f"figures/{prefix}_power_by_size.png",
            "monthly": f"figures/{prefix}_alert_rate_by_month.png",
        }
    tz = cfg.region(region_id).local_tz
    text = render(
        versions,
        cfg,
        tz,
        gates,
        live_refusals(engine, cfg, region_id),
        unchanged,
        explain_runs,
        figures,
    )
    return HoldoutReport(text, versions)


def write_report(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    reports = cfg.resolve(cfg.paths.reports_dir)
    report = build_report(engine, cfg, region_id, reports / "figures")
    path = reports / REPORT_NAME
    path.write_text(report.text, encoding="utf-8")
    return path

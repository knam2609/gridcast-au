"""Generate ``reports/m2_holdout.md`` (+ figures): the M2 holdout, scored once (step 6).

Nothing in the report is typed by hand. Every number is read from the stored holdout
batches (``gridcast evaluate-holdout``), run 2's development batches, AEMO's archived
forecasts and the daily-report match check (``gridcast ingest-daily``), or measured when the
report is built (the leakage property tests are run by this command).

* **S5** (m2_plan.md §3) is judged from what is stored: one batch per evaluation set for the
  frozen config hash, all from one run whose reproduction check passed. While they exist,
  ``evaluate-holdout`` refuses to run again.
* **Replication (step 6, decision 4):** S1–S4 are recomputed on the holdout exactly as
  written, with the same code as the development report, and reported as *holds* or *does
  not hold*. They do not change the development verdicts.
* The holdout part of the AEMO benchmark uses the step-5 alignments and bootstrap.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.evaluation import aemo_benchmark as aemo
from gridcast.evaluation.holdout import HOLDOUT_SETS, stored_holdout_batches
from gridcast.evaluation.m2_report import (
    HEADLINE,
    MAIN_TABLE,
    ORACLE,
    S1_FIRST_STEP,
    SEGMENT_MODELS,
    WEATHER_FREE,
    Criterion,
    DelayCheck,
    LeakageOutcome,
    Pairs,
    _failure_section,
    _line_chart,
    _weather_value_table,
    band_jumps,
    baselines_of,
    first_better,
    group_means,
    load_pairs,
    pooled,
    run_leakage_tests,
    selection_table,
    success_criteria,
)
from gridcast.evaluation.m2_results import (
    _load_batch,
    load_evaluation_set,
    retrain_records,
    run_config,
)
from gridcast.evaluation.report import (
    TargetResults,
    _fmt,
    _md_table,
    horizon_label,
    horizon_table,
    model_label,
)
from gridcast.evaluation.segments import describe_segments, describe_weather_segments
from gridcast.features.inputs import load_feature_builder
from gridcast.forecasting.ml_backtest import ml_config_hash
from gridcast.ingestion.daily import SOURCE as DAILY_SOURCE
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

FIGURE_LEAD = "m2_holdout_mae_by_lead.png"
FIGURE_AEMO = "m2_holdout_aemo_mae_by_lead.png"
MAIN_PLOT = ("daily_naive", "persistence", "lgbm_A", "ridge_C", "lgbm_C")
PERIOD_MODELS = (
    "lgbm_C",
    "lgbm_A",
    "lgbm_D",
    "ridge_C",
    "persistence",
    "daily_naive",
    "weekly_naive",
)
QUARTER_MODELS = ("lgbm_A", "lgbm_C", "lgbm_D", "ridge_C", "daily_naive")
PLUS_MODELS = (
    "lgbm_Cplus",
    "lgbm_Dplus",
    "lgbm_C",
    "ridge_Cplus",
    "ridge_Dplus",
    "ridge_C",
    "persistence",
    "daily_naive",
    "weekly_naive",
)
_CHECK_FILE = re.compile(r"^check_(\d{4}-\d{2})\.json$")


# --------------------------------------------------------------------------- loading


def load_holdout_set(engine: Engine, region_id: str, key: str, config_hash: str) -> TargetResults:
    """The stored holdout batch of evaluation set ``key`` for ``config_hash``."""
    with engine.connect() as conn:
        batch = conn.execute(
            text(
                """
                SELECT batch_id FROM model_runs
                WHERE region_id = :r AND target = 'demand' AND status = 'success'
                  AND config->>'evaluation_set' = :k AND config->>'config_hash' = :h
                ORDER BY created_at DESC LIMIT 1
                """
            ),
            {"r": region_id, "k": HOLDOUT_SETS[key], "h": config_hash},
        ).scalar()
    if batch is None:
        raise LookupError(
            f"no {HOLDOUT_SETS[key]} batch for config {config_hash}; "
            "run `gridcast evaluate-holdout`"
        )
    return _load_batch(engine, str(batch))


@dataclass(slots=True)
class HoldoutSets:
    main: TargetResults
    plus: TargetResults | None
    delay: TargetResults | None

    def items(self) -> list[tuple[str, TargetResults]]:
        return [
            (k, r)
            for k, r in (("main", self.main), ("plus", self.plus), ("delay", self.delay))
            if r is not None
        ]


def _optional(engine: Engine, region_id: str, key: str, h: str) -> TargetResults | None:
    try:
        return load_holdout_set(engine, region_id, key, h)
    except LookupError:
        return None


def _set_config(res: TargetResults) -> dict[str, Any]:
    return run_config(res, str(res.models[0]))


# --------------------------------------------------------------------------- criteria


def s5_criterion(sets: HoldoutSets, stored: Sequence[str], frozen: str) -> Criterion:
    """S5 from what is stored: one run, one batch per set, reproduction passed."""
    configs = [_set_config(res) for _, res in sets.items()]
    shas = sorted({str(c["code_sha"]) for c in configs})
    batches = sorted(str(res.batch_id) for _, res in sets.items())
    reproduced = all(bool(c["reproduction"]["passed"]) for c in configs)
    frozen_ok = all(str(c["config_hash"]) == frozen for c in configs)
    one_run = len(shas) == 1 and sorted(stored) == batches
    sha_text = ", ".join(f"`{s[:7]}`" for s in shas)
    return Criterion(
        "S5",
        "holdout scored by a single `gridcast evaluate-holdout` run against the frozen model "
        "config; the command refuses to run again for the same config hash",
        f"{len(stored)} holdout batch(es) stored for config `{frozen}`, "
        f"{len(configs)} evaluation set(s), from run commit(s) {sha_text}; reproduction check "
        f"{'passed' if reproduced else 'FAILED'} for every set; `evaluate-holdout` refuses "
        "while these batches exist",
        bool(one_run and reproduced and frozen_ok),
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
    headers = ["#", "Pass if", "Development (run 2)", "Dev", "Holdout", "Holdout"]
    return _md_table(headers, rows, "llllll")


# --------------------------------------------------------------------------- tables


def envelope(table: pd.DataFrame, baselines: Sequence[str]) -> pd.Series:
    """Best baseline's MAE at each lead time."""
    return table[list(baselines)].min(axis=1)


def period_table(
    dev: TargetResults,
    hold: TargetResults,
    models: Sequence[str],
    baselines: Sequence[str],
) -> str:
    """Pooled MAE per model, development (run 2) vs holdout, and the change."""
    dp, hp = pooled(dev, "dev"), pooled(hold, "holdout")
    rows = []
    for m in models:
        if m in dp and m in hp:
            rows.append(
                [model_label(m), _fmt(dp[m]), _fmt(hp[m]), f"{100 * (hp[m] / dp[m] - 1):+.1f}%"]
            )
    de = float(envelope(dev.by_horizon("dev", "mae"), baselines).mean())
    he = float(envelope(hold.by_horizon("holdout", "mae"), baselines).mean())
    rows.append(
        ["Best baseline at each lead (mean)", _fmt(de), _fmt(he), f"{100 * (he / de - 1):+.1f}%"]
    )
    return _md_table(["Model", "Development", "Holdout", "Change"], rows)


def lead_group_table(
    dev_table: pd.DataFrame, hold_table: pd.DataFrame, baselines: Sequence[str]
) -> str:
    """Per lead-time range: LightGBM C and the best baseline, development vs holdout."""
    frames = {}
    for name, table in (("dev", dev_table), ("hold", hold_table)):
        t = table.copy()
        t["envelope"] = envelope(table, baselines)
        frames[name] = group_means(t)
    rows = []
    for group in frames["dev"].index:
        d, h = frames["dev"].loc[group], frames["hold"].loc[group]
        rows.append(
            [
                str(group),
                _fmt(d[HEADLINE]),
                _fmt(h[HEADLINE]),
                _fmt(d["envelope"]),
                _fmt(h["envelope"]),
                f"{100 * (1 - d[HEADLINE] / d['envelope']):.0f}%",
                f"{100 * (1 - h[HEADLINE] / h['envelope']):.0f}%",
            ]
        )
    headers = [
        "Lead time",
        "C dev",
        "C holdout",
        "Best baseline dev",
        "Best baseline holdout",
        "C below best, dev",
        "C below best, holdout",
    ]
    return _md_table(headers, rows)


def quarterly_errors(pairs: Pairs, models: Sequence[str]) -> pd.DataFrame:
    """Pooled MAE per NEM-time quarter of the issue time (all lead times)."""
    nem = pairs.origin_times.tz_convert(NEM_TZ)
    quarter = np.asarray([f"{t.year} Q{(t.month - 1) // 3 + 1}" for t in nem])
    rows = {}
    for q in sorted(set(quarter)):
        sel = quarter == q
        mask = pairs.mask[sel]
        y = pairs.actual[sel][mask]
        row = {m: float(np.mean(np.abs(pairs.predictions[m][sel][mask] - y))) for m in models}
        row["n"] = int(mask.sum())
        rows[q] = row
    return pd.DataFrame(rows).T


def _daily_check(cfg: AppConfig) -> dict[str, Any] | None:
    """The latest daily-report match check that loaded something (``ingest-daily``)."""
    folder = cfg.data_path(cfg.paths.interim_dir) / DAILY_SOURCE
    if not folder.is_dir():
        return None
    found = []
    for name in sorted(p.name for p in folder.iterdir()):
        if _CHECK_FILE.match(name):  # never AppleDouble ``._`` files on exFAT
            payload = json.loads((folder / name).read_text())
            if payload.get("loaded"):
                found.append(payload)
    return found[-1] if found else None


def _daily_rows(engine: Engine, region_id: str, first: pd.Timestamp, last: pd.Timestamp) -> int:
    """Market intervals in [first, last] loaded from the daily reports."""
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    """
                    SELECT count(*) FROM market_intervals m
                    JOIN ingestion_runs r ON r.run_id = m.ingestion_run_id
                    WHERE m.region_id = :reg AND r.source = :src
                      AND m.interval_end BETWEEN :a AND :b
                    """
                ),
                {
                    "reg": region_id,
                    "src": DAILY_SOURCE,
                    "a": first.to_pydatetime(),
                    "b": last.to_pydatetime(),
                },
            ).scalar_one()
        )


def _provenance(
    cfg: AppConfig,
    sets: HoldoutSets,
    dev: dict[str, TargetResults],
    leakage: LeakageOutcome,
    region_id: str,
    report_sha: str | None,
    daily_rows: int,
    check: dict[str, Any] | None,
) -> str:
    rc = run_config(sets.main, HEADLINE)
    all_runs = pd.concat([r.runs for _, r in sets.items()])
    stored_shas = sorted({str(s) for s in all_runs["git_sha"]})
    current = ml_config_hash(cfg)
    excluded = rc["total_pairs"] - rc["scored_pairs"]
    first = pd.Timestamp(rc["first_issue"]).tz_convert(NEM_TZ)
    last = pd.Timestamp(rc["last_issue"]).tz_convert(NEM_TZ)
    if daily_rows and check is not None:
        cols = {c["column"]: c for c in check["check"]["columns"]}
        demand = cols["total_demand_mw"]
        daily_text = (
            f"{daily_rows:,} target intervals ({', '.join(check['loaded'])}) from NEMWeb's daily "
            f"dispatch reports, loaded after a match check against MMSDM over "
            f"{check['check']['month']}: {demand['compared']:,} intervals compared, demand "
            f"different at {demand['differing']:,} (`docs/data_sources.md` §7)"
        )
    elif daily_rows:
        daily_text = f"{daily_rows:,} target intervals from the daily reports (no check file)"
    else:
        daily_text = "none: every target interval comes from MMSDM"
    rows = [
        ["Region", region_id],
        [
            "Holdout batches",
            ", ".join(f"{HOLDOUT_SETS[k]} `{r.batch_id}`" for k, r in sets.items()),
        ],
        [
            "Run commit (recorded at the start of the run)",
            ", ".join(f"`{s}`" for s in stored_shas),
        ],
        [
            "ML config hash",
            f"`{rc['config_hash']}` = `backtest.frozen_ml_config_hash`"
            + (
                " (matches the current config)"
                if current == rc["config_hash"]
                else f" (**differs** from the current config `{current}`)"
            ),
        ],
        ["Model and feature code frozen at", f"`{rc['frozen_commit']}` (unchanged at the run)"],
        [
            "Holdout issue times (NEM time)",
            f"{first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M}, {rc['n_origins']:,} "
            f"(every {5 * rc['setup']['issue_every_steps']} min)",
        ],
        [
            "Retrains forecasting the holdout",
            f"{len(rc['retrains'])}: {', '.join(rc['retrains'])} (the first is the last "
            "development retrain)",
        ],
        [
            "Scored (issue time, lead) pairs, every model",
            f"{rc['scored_pairs']:,} of {rc['total_pairs']:,} "
            f"({excluded:,} excluded, {100 * excluded / rc['total_pairs']:.2f}%)",
        ],
        [
            "Training rows at the last retrain (shared by A, C, D)",
            f"{rc['shared_train_rows']:,}",
        ],
        [
            "Development reference (run 2)",
            ", ".join(f"{k} `{r.batch_id}`" for k, r in dev.items()),
        ],
        ["Market data after the MMSDM archive", daily_text],
        [
            "Leakage property tests (run by this command)",
            f"{leakage.passed} passed, {leakage.failed} failed",
        ],
        ["Report generated from commit", f"`{report_sha}`"],
    ]
    return _md_table(["Item", "Value"], rows, "ll")


def _reproduction_table(sets: HoldoutSets) -> str:
    rows = []
    for key, res in sets.items():
        rep = _set_config(res)["reproduction"]
        rows.append(
            [
                HOLDOUT_SETS[key],
                f"`{rep['reference_batch']}`",
                rep["retrain"],
                str(len(rep["models"])),
                f"{rep['issue_times']:,}",
                f"{rep['pairs']:,}",
                str(rep["fits"]),
                "exact" if rep["passed"] else "**FAILED**",
            ]
        )
    headers = [
        "Set",
        "Run 2 batch",
        "Retrain",
        "Models",
        "Dev issue times",
        "Forecasts per model",
        "Fits",
        "Result",
    ]
    return _md_table(headers, rows, "lllrrrrl")


# --------------------------------------------------------------------------- AEMO


def _aemo_section(
    engine: Engine, cfg: AppConfig, region_id: str, main: TargetResults, figures: Path
) -> list[str]:
    b = aemo.load_benchmark(engine, cfg, region_id, "holdout", batch=main)
    c = aemo.compare(b, cfg.ml.seed)
    dev = aemo.compare(aemo.load_benchmark(engine, cfg, region_id), cfg.ml.seed)
    first, end = aemo.window(cfg, "holdout")
    last_day = (end - pd.Timedelta(days=1)).tz_convert(NEM_TZ)
    title = (
        f"VIC1 demand, {first.tz_convert(NEM_TZ):%b %Y} → {last_day:%b %Y} (holdout): "
        "30-minute periods, same pairs"
    )
    aemo._figure(c.mae30, figures / FIGURE_AEMO, title)

    names30 = [aemo.MATCHED, aemo.STRICT, HEADLINE, "lgbm_A", "lgbm_D", "ridge_C"]
    leads30 = list(c.mae30.index)
    win_m = int((c.mae30[HEADLINE] < c.mae30[aemo.MATCHED]).sum())
    win_s = int((c.mae30[HEADLINE] < c.mae30[aemo.STRICT]).sum())
    loses_m = aemo.lead_runs(list(c.mae30[HEADLINE] >= c.mae30[aemo.MATCHED]), leads30)
    leads5 = list(c.mae5.index)
    win5 = int((c.mae5[HEADLINE] < c.mae5[aemo.P5]).sum())
    loses5 = aemo.lead_runs(list(c.mae5[HEADLINE] >= c.mae5[aemo.P5]), leads5)
    covered = [k for k in leads30 if np.isfinite(b.aemo30[aemo.MATCHED][:, k - 1]).mean() >= 0.999]
    full = max(covered) if covered else 0

    diff_rows = []
    for d_dev, d_hold in zip([*dev.diffs30, dev.diff5], [*c.diffs30, c.diff5], strict=True):
        diff_rows.append(
            [
                aemo.label(d_hold.a),
                aemo.label(d_hold.b),
                f"{d_dev.diff:+,.1f} ({d_dev.lo:+,.1f} to {d_dev.hi:+,.1f})",
                f"{d_hold.diff:+,.1f} ({d_hold.lo:+,.1f} to {d_hold.hi:+,.1f})",
                f"{d_hold.n_pairs:,}",
                f"{d_hold.n_days:,}",
            ]
        )
    p5_rows = [
        [
            aemo.step_label(h),
            _fmt(aemo._at(c.mae5, h, aemo.P5)),
            _fmt(aemo._at(c.mae5, h, HEADLINE)),
            _fmt(aemo._at(c.mae5, h, "persistence")),
            f"{int(aemo._at(c.mae5, h, 'n')):,}",
        ]
        for h in leads5
    ]
    p5_rows.append(
        [
            "**All (pooled)**",
            f"**{_fmt(c.pooled5[aemo.P5])}**",
            f"**{_fmt(c.pooled5[HEADLINE])}**",
            f"**{_fmt(c.pooled5['persistence'])}**",
            "",
        ]
    )
    return [
        "## AEMO benchmark, holdout part\n",
        "The step-5 comparison (`reports/m2_aemo.md`) repeated on the holdout issue times, with "
        "the same alignments (matched is the headline; strict is the latest run published by "
        "the issue time), the same pairs rule and the same day-block bootstrap "
        f"({aemo.BOOTSTRAP_DRAWS:,} draws, seed `ml.seed`). AEMO's forecasts are a benchmark "
        "only, not pass/fail. "
        + (
            f"AEMO (matched) covers every issue time up to {aemo.period_label(full)} ahead; "
            "longer leads cover fewer issue times.\n"
            if full < len(leads30)
            else "AEMO (matched) covers every issue time at every lead.\n"
        ),
        f"![AEMO vs LightGBM C, holdout](figures/{FIGURE_AEMO})\n",
        "### MAE (MW) of 30-minute mean demand, holdout\n",
        aemo._mae_table(c.mae30, names30, c.pooled30),
        "",
        f"- {aemo.label(HEADLINE)} has a lower MAE than AEMO (matched) at {win_m} of "
        f"{len(leads30)} leads; AEMO (matched) is as good or better at: "
        f"{aemo._ranges_text(loses_m, aemo.period_label)}.",
        f"- Against AEMO (strict), {aemo.label(HEADLINE)} is lower at {win_s} of "
        f"{len(leads30)} leads.",
        "",
        "### Pooled difference (MW, A − B, 95% day-block interval), development vs holdout\n",
        _md_table(
            [
                "Forecast A",
                "Forecast B",
                "Development",
                "Holdout",
                "Holdout pairs",
                "Holdout days",
            ],
            diff_rows,
            "llrrrr",
        ),
        "",
        *[
            f"- Holdout, {aemo.label(d.a)} vs {aemo.label(d.b)}: {aemo._verdict(d)}."
            for d in (*c.diffs30, c.diff5)
        ],
        "",
        "### By quarter of the issue time (all leads pooled, MAE in MW)\n",
        aemo._quarter_table(b, c.mask30),
        "",
        "### 5-minute pre-dispatch vs our forecasts (leads 5–55 minutes), holdout\n",
        _md_table(
            [
                "Lead",
                aemo.label(aemo.P5),
                aemo.label(HEADLINE),
                aemo.label("persistence"),
                "Pairs",
            ],
            p5_rows,
        ),
        "",
        f"- {aemo.label(HEADLINE)} has a lower MAE than AEMO 5-minute pre-dispatch at {win5} of "
        f"{len(leads5)} leads; AEMO is as good or better at: "
        f"{aemo._ranges_text(loses5, aemo.step_label)}.",
        "",
    ]


# --------------------------------------------------------------------------- report


def build_report(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    leakage: LeakageOutcome | None = None,
    include_aemo: bool = True,
) -> str:
    report_sha = git_sha()  # before any figure is written, which would make the tree dirty
    frozen = cfg.backtest.frozen_ml_config_hash
    if frozen is None:
        raise LookupError("no frozen ML config (backtest.frozen_ml_config_hash)")
    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    figures = reports_dir / "figures"
    data_root = cfg.data_root()
    sets = HoldoutSets(
        load_holdout_set(engine, region_id, "main", frozen),
        _optional(engine, region_id, "plus", frozen),
        _optional(engine, region_id, "delay", frozen),
    )
    main = sets.main
    rc = run_config(main, HEADLINE)
    dev: dict[str, TargetResults] = {}
    for key, _ in sets.items():
        res = load_evaluation_set(engine, region_id, key)
        if str(run_config(res, str(res.models[0]))["config_hash"]) != frozen:
            raise LookupError(f"the latest development '{key}' batch is not the frozen config's")
        dev[key] = res
    leakage = leakage or run_leakage_tests()
    stored = stored_holdout_batches(engine, region_id, frozen)

    table = main.by_horizon("holdout", "mae")
    dev_table = dev["main"].by_horizon("dev", "mae")
    baselines = baselines_of(main)
    s5 = s5_criterion(sets, stored, frozen)
    dev_criteria = success_criteria(dev_table, pooled(dev["main"], "dev"), baselines, leakage)
    hold_criteria = success_criteria(table, pooled(main, "holdout"), baselines, leakage)
    holds = sum(c.passed for c in hold_criteria)
    hold_pooled = pooled(main, "holdout")
    dev_pooled = pooled(dev["main"], "dev")

    bands = [(int(lo), int(hi)) for lo, hi in rc["lead_bands"]]
    boundaries = [hi for _, hi in bands[:-1]]
    boundary_hours = [b / 12 for b in boundaries]
    _line_chart(
        table,
        MAIN_PLOT,
        figures / FIGURE_LEAD,
        "Victoria demand: MAE by lead time (holdout, scored once)",
        24,
        boundary_hours,
    )
    horizons = sorted(
        {*cfg.evaluation.horizons_reported, *boundaries, *[b + 1 for b in boundaries]}
    )
    builder = load_feature_builder(engine, cfg, region_id)
    pairs = load_pairs(main, data_root, builder.ctx.times)
    if int(pairs.mask.sum()) != int(rc["scored_pairs"]):
        raise ValueError("stored predictions do not reproduce the stored scored-pair count")
    delay = (
        DelayCheck.of(sets.delay, float(rc["publication_delay_hours"]), "holdout")
        if sets.delay is not None
        else None
    )
    cross = first_better(table, HEADLINE, "persistence")
    jumps, typical = band_jumps(table, HEADLINE, boundaries)
    quarters = quarterly_errors(pairs, [m for m in QUARTER_MODELS if m in main.models])
    c_dev, c_hold = dev_pooled[HEADLINE], hold_pooled[HEADLINE]
    env_dev = float(envelope(dev_table, baselines).mean())
    env_hold = float(envelope(table, baselines).mean())
    holdout_first = pd.Timestamp(rc["first_issue"]).tz_convert(NEM_TZ)
    holdout_last = pd.Timestamp(rc["last_issue"]).tz_convert(NEM_TZ)
    s1_label = horizon_label(S1_FIRST_STEP)
    horizon = pd.Timedelta(minutes=5 * int(rc["setup"]["horizon_steps"]))
    daily_rows = _daily_rows(
        engine,
        region_id,
        pd.Timestamp(rc["first_issue"]),
        pd.Timestamp(rc["last_issue"]) + horizon,
    )
    daily = _daily_check(cfg)
    drc = run_config(dev["main"], HEADLINE)
    dev_first = pd.Timestamp(drc["first_issue"]).tz_convert(NEM_TZ)
    dev_last = pd.Timestamp(drc["last_issue"]).tz_convert(NEM_TZ)

    out: list[str] = [
        "# Milestone 2: the holdout, scored once\n",
        "> **Generated** by `gridcast report-holdout` on "
        f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M UTC} from the stored holdout run "
        f"(issue times {holdout_first:%Y-%m-%d} → {holdout_last:%Y-%m-%d}, NEM time). Do not "
        "edit by hand. The model config was frozen before the AEMO benchmark and before any "
        "holdout scoring; this is the unbiased estimate of how the recipe performs.\n",
        "## Summary\n",
        (
            "- **S5 passes:** the holdout was scored by one `evaluate-holdout` run of the "
            "frozen config, after it reproduced run 2's development forecasts exactly."
            if s5.passed
            else f"- **S5 FAILS:** {s5.measured}."
        ),
        f"- **{holds} of {len(hold_criteria)} development criteria hold on the holdout** "
        "(replication, reported, not pass/fail).",
        f"- LightGBM C's pooled MAE is {_fmt(c_hold)} MW on the holdout against "
        f"{_fmt(c_dev)} MW in development ({100 * (c_hold / c_dev - 1):+.1f}%); the best "
        f"baseline's mean MAE moved from {_fmt(env_dev)} to {_fmt(env_hold)} MW "
        f"({100 * (env_hold / env_dev - 1):+.1f}%).",
        "",
        "## Provenance\n",
        _provenance(cfg, sets, dev, leakage, region_id, report_sha, daily_rows, daily),
        "",
        "## S5: holdout used once\n",
        _md_table(
            ["#", "Pass if", "Measured", "Result"],
            [[s5.key, s5.rule, s5.measured, "**PASS**" if s5.passed else "**FAIL**"]],
            "lllc",
        ),
        "",
        "**Reproduction check (step 6, decision 3).** The run started at the last development "
        "retrain and forecast its whole fold. Before any holdout pair was scored, its forecasts "
        "for the development issue times had to equal run 2's stored forecasts exactly "
        "(float32, NaN in the same places), and its fits had to match run 2's stored fit "
        "records (setting, tree count, rows, every candidate's early-stopping MAE):\n",
        _reproduction_table(sets),
        "",
        "## Replication of S1–S4 on the holdout\n",
        "Pre-registered on 2026-11-09, before the holdout run (m2_plan.md step 6, decision 4): "
        "S1–S4 are recomputed exactly as written in §3, by the same code as the development "
        "report, on the holdout pairs. They are reported, not pass/fail, and do not change the "
        "development verdicts. *Best baseline* is the lowest MAE among "
        f"{', '.join(model_label(b).lower() for b in baselines)} at each lead time.\n",
        replication_table(dev_criteria, hold_criteria),
        f"\n**{holds} of {len(hold_criteria)} hold on the holdout.**\n",
        "## Development vs holdout\n",
        "Pooled MAE (MW) over every lead time. The periods differ (development: issue times "
        f"{dev_first:%b %Y} → {dev_last:%b %Y}; holdout: {holdout_first:%b %Y} → "
        f"{holdout_last:%b %Y}), so a change mixes how well the recipe generalises with how "
        "hard the period was; the baselines' change shows the latter.\n",
        period_table(dev["main"], main, [m for m in PERIOD_MODELS if m in main.models], baselines),
        "",
        "Mean of the per-lead MAE (MW) within each lead-time range:\n",
        lead_group_table(dev_table, table, baselines),
        "",
        "## MAE by lead time (holdout)\n",
        f"![MAE by lead time, holdout](figures/{FIGURE_LEAD})\n",
        "### MAE (MW)\n",
        horizon_table(
            main, "holdout", horizons, "mae", [m for m in MAIN_TABLE if m in main.models]
        ),
        "\n### RMSE (MW)\n",
        horizon_table(
            main, "holdout", horizons, "rmse", [m for m in MAIN_TABLE if m in main.models]
        ),
        "\n### Bias (MW, forecast − actual)\n",
        horizon_table(
            main, "holdout", horizons, "bias", [m for m in MAIN_TABLE if m in main.models]
        ),
        "",
        "## Short lead times and the band boundaries (holdout)\n",
        f"- At {horizon_label(1)}, LightGBM C's MAE is {_fmt(float(table[HEADLINE].iloc[0]))} "
        f"MW against {_fmt(float(table['persistence'].iloc[0]))} MW for persistence; "
        + (
            "LightGBM C never beats persistence."
            if cross is None
            else f"LightGBM C is better from {horizon_label(cross)} on."
            if bool((table.loc[cross:, HEADLINE] < table.loc[cross:, "persistence"]).all())
            else f"LightGBM C first beats it at {horizon_label(cross)}."
        ),
        *[
            f"- Between {horizon_label(b)} and {horizon_label(b + 1)} (a band boundary), "
            f"LightGBM C's MAE changes by {j:+,.1f} MW, against a median step of "
            f"{typical:,.1f} MW between neighbouring lead times elsewhere."
            for b, j in zip(boundaries, jumps, strict=True)
        ],
        "",
        "## Weather value by lead time (holdout)\n",
        "Mean of the per-lead MAE within each lead-time range (MW). The delay column comes "
        "from the delay batch, on its own common pairs.\n",
        _weather_value_table(table, delay),
        "",
    ]
    if sets.plus is not None:
        prc = run_config(sets.plus, "lgbm_Cplus")
        out += [
            "## C⁺ and D⁺: all forecast weather variables (holdout, reported, not pass/fail)\n",
            f"C⁺ and D⁺ next to headline C on their common pairs ({prc['scored_pairs']:,}).\n",
            horizon_table(
                sets.plus,
                "holdout",
                horizons,
                "mae",
                [m for m in PLUS_MODELS if m in sets.plus.models],
            ),
            "",
        ]
    if sets.delay is not None and delay is not None:
        out += [
            "## Publication-delay sensitivity (holdout, never the headline)\n",
            f"C rebuilt as if every GFS run were published {delay.hours:g} h after its run "
            f"time (headline: {delay.headline_hours:g} h), with headline C's chosen settings at "
            "every retrain.\n",
            horizon_table(sets.delay, "holdout", horizons, "mae"),
            "",
        ]
    out += [
        "## Errors by quarter (holdout)\n",
        "Pooled MAE (MW) by NEM-time quarter of the issue time.\n",
        _md_table(
            [
                "Quarter",
                "Pairs",
                *[model_label(m) for m in quarters.columns if m != "n"],
                "C − D",
            ],
            [
                [
                    str(q),
                    f"{int(r['n']):,}",
                    *[_fmt(r[m]) for m in quarters.columns if m != "n"],
                    f"{r[HEADLINE] - r[ORACLE]:+.1f}",
                ]
                for q, r in quarters.iterrows()
            ],
        ),
        "",
        "## Model selection at the holdout retrains\n",
        "LightGBM C, per retrain and lead band: the winning setting and the best early-stopping "
        "MAE each target form reached. Retrains inside the holdout select on their own last "
        f"{rc['ml']['early_stopping_weeks']} weeks, which lie in the holdout period, as an "
        "operational retrain would.\n",
        selection_table(retrain_records(main, HEADLINE)),
        "",
    ]
    seg_desc = {**describe_segments(cfg.evaluation), **describe_weather_segments(cfg.evaluation)}
    seg_models = [m for m in SEGMENT_MODELS if m in main.models]
    seg_rows = []
    for seg, text_ in seg_desc.items():
        values = [main.value(m, "holdout", seg, 0, "mae") for m in seg_models]
        finite = [v for v in values if pd.notna(v)]
        best = min(finite) if finite else None
        seg_rows.append(
            [
                text_,
                f"{main.n('holdout', seg, 0):,}",
                *[f"**{_fmt(v)}**" if v == best else _fmt(v) for v in values],
            ]
        )
    out += [
        "## MAE (MW) by segment, all lead times pooled (holdout)\n",
        "Segments are defined by what actually happened (*ex-post*).\n",
        _md_table(["Segment", "Pairs", *[model_label(m) for m in seg_models]], seg_rows),
        "",
        *_failure_section(cfg, engine, region_id, pairs, builder),
    ]
    if include_aemo:
        out += _aemo_section(engine, cfg, region_id, main, figures)
    out += [
        "## Notes and limitations\n",
        "- **One look.** The holdout is scored once (S5). A bug found later is disclosed "
        "and its effect reported; the holdout is not re-run to fix it.\n"
        "- **Development numbers are slightly optimistic.** The development period was "
        "looked at twice (two development runs, `reports/m2_demand.md`); these holdout numbers "
        "are not.\n"
        "- **Retrains inside the holdout** select their settings on their own last weeks, "
        "which lie in the holdout period. An operational retrain would do the same; no "
        "forecast uses data after its issue time.\n"
        "- **Split by issue time** (as in M1): development issue times on the last day before "
        "the holdout have targets on its first day. Those actuals were used to score the "
        "development period, never to fit a model.\n"
        + (
            f"- **{daily_rows:,} target intervals** come from NEMWeb's daily dispatch reports, "
            "because the monthly MMSDM archive for that month was not yet published (see "
            "*Provenance*).\n"
            if daily_rows
            else ""
        )
        + f"- S1–S4 replications judge LightGBM C only; ridge, C⁺/D⁺, the delay check, AEMO and "
        f"lead times under {s1_label} are reported, not pass/fail.\n"
        f"- {model_label(WEATHER_FREE)} and {model_label(ORACLE)} use the same retrains and "
        "pairs as C, so C − A is the value of forecast temperature and C − D the cost of its "
        "error.\n",
    ]
    return "\n".join(out)


def write_report(
    engine: Engine, cfg: AppConfig, region_id: str, leakage: LeakageOutcome | None = None
) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / "m2_holdout.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id, leakage))
    return path

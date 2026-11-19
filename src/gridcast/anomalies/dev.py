"""The development run of the frozen anomaly recipe (m4_plan.md step 4): ``gridcast anomalies-dev``.

**Preconditions** (:class:`DevRefused`; nothing is computed): the freeze is recorded and
holds (the anomaly config hash equals ``anomalies.frozen_config_hash`` and no file under
:data:`~gridcast.anomalies.freeze.FROZEN_PATHS` changed since ``anomalies.frozen_commit``),
the working tree is clean, and no successful development run is stored for the hash.
Development is scored once (§2.1).

**The run** reads data only up to the development end (NEM midnight of
``backtest.holdout_start``):

1. the market series and the development band (:func:`~gridcast.anomalies.inputs.split_band`),
   with the stored actuals checked against market demand;
2. every detector, tier and event (:func:`~gridcast.anomalies.run.detect`);
3. the injections (§2.9, 1,000 shared starts per detector);
4. A5 on real data: the detection repeated on the series truncated at each seeded cut
   (:mod:`~gridcast.anomalies.recipe`), and the forests' training ranges;
5. evidence and summaries for every event, with SHAP at the demand peaks behind E1's gate.
   A gate failure stops the run before anything is stored (explanations withheld, §2.6).

**Storage:** one ``model_runs`` row (kind ``anomaly``, target ``market``), parquet under
``<data root>/anomalies/<batch>/`` (per-interval scores, p-values, window sizes, tiers and
scored-period flags; injections; forests; cuts), and the events with their evidence and
summaries in ``anomalies``. The verdicts are computed by ``gridcast report-m4`` from these.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Connection, Engine, text

from gridcast.anomalies.detectors import LINES, Recipe
from gridcast.anomalies.evidence import banned_in, missing_fields
from gridcast.anomalies.evidence_inputs import EvidenceRun, run_evidence
from gridcast.anomalies.freeze import FreezeState, freeze_state
from gridcast.anomalies.injection import run_injections
from gridcast.anomalies.inputs import check_band_actuals, load_market_series, split_band
from gridcast.anomalies.recipe import (
    cut_indices,
    dev_split,
    differences,
    forest_leaks,
    truncated_detection,
)
from gridcast.anomalies.run import Detection, Split, detect
from gridcast.anomalies.series import MarketSeries
from gridcast.config import AppConfig
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.logs import get_logger

log = get_logger(__name__)

MODEL_NAME = "anomalies"
FILES = ("intervals.parquet", "injections.parquet", "forests.parquet", "cuts.parquet")

EXISTING_SQL = """
SELECT run_id, batch_id FROM model_runs
WHERE model_kind = 'anomaly' AND region_id = :region AND status = 'success'
  AND config->>'split' = :split AND config->>'anomaly_config_hash' = :hash
"""


class DevRefused(RuntimeError):
    """A precondition of the development run is not met; nothing was run."""


def existing_runs(engine: Engine, region_id: str, split: str, config_hash: str) -> list[tuple]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(EXISTING_SQL), {"region": region_id, "split": split, "hash": config_hash}
        ).all()
    return [tuple(r) for r in rows]


def check_preconditions(engine: Engine, cfg: AppConfig, region_id: str) -> FreezeState:
    state = freeze_state(cfg)
    problems = list(state.problems)
    if state.sha is None:
        problems.append("not a git checkout")
    if state.dirty:
        problems.append("the working tree has uncommitted changes to tracked files")
    stored = existing_runs(engine, region_id, "dev", state.config_hash)
    if stored:
        problems.append(
            f"a development run is already stored for hash {state.config_hash} "
            f"(run {stored[0][0]}, batch {stored[0][1]}): development is scored once"
        )
    if problems:
        raise DevRefused("; ".join(problems))
    return state


# --------------------------------------------------------------------------- the run


@dataclass(slots=True)
class CutCheck:
    index: int
    end: pd.Timestamp
    differences: list[str]
    seconds: float


@dataclass(slots=True)
class DevOutputs:
    split: Split
    series: MarketSeries
    detection: Detection
    injections: pd.DataFrame
    cuts: list[CutCheck]
    forest_leaks: list[str]
    evidence: EvidenceRun
    band_issue_times: int
    band_valid_pairs: int
    checked_actuals: int
    seconds: dict[str, float] = field(default_factory=dict)


def compute_dev(
    engine: Engine, cfg: AppConfig, region_id: str, recipe: Recipe | None = None
) -> DevOutputs:
    """Everything the development run stores (no writes); ``recipe`` None is version 1."""
    a = cfg.anomalies
    split = dev_split(cfg)
    last = nem_midnight_utc(str(cfg.backtest.holdout_start))
    secs: dict[str, float] = {}

    t = time.perf_counter()
    series = load_market_series(engine, cfg, region_id, last_end=last)
    if series.ends[-1] != last:
        raise ValueError(f"the market series ends at {series.ends[-1]}, not at {last}")
    band, actual = split_band(cfg, "dev")
    band = band.issued_until(last)
    actual = actual[: len(band.issue_times)]
    checked = check_band_actuals(band, actual, series)
    secs["load"] = time.perf_counter() - t
    log.info("anomalies.dev.loaded", intervals=len(series), issue_times=len(band.issue_times))

    t = time.perf_counter()
    det = detect(series, band, split, a, recipe=recipe)
    secs["detect"] = time.perf_counter() - t
    log.info("anomalies.dev.detected", events=len(det.events), seconds=round(secs["detect"], 1))

    t = time.perf_counter()
    inj = run_injections(series, det.run, split, a)
    secs["inject"] = time.perf_counter() - t
    log.info("anomalies.dev.injected", rows=len(inj), seconds=round(secs["inject"], 1))

    cuts = []
    for c in cut_indices(series.ends, split, a):
        t = time.perf_counter()
        cut_run = truncated_detection(series, band, split, a, int(c), recipe=det.run.recipe)
        diff = differences(det, cut_run, int(c))
        cuts.append(CutCheck(int(c), series.ends[int(c)], diff, time.perf_counter() - t))
        log.info("anomalies.dev.cut", cut=str(series.ends[int(c)]), differences=len(diff))
    secs["cuts"] = sum(cc.seconds for cc in cuts)
    leaks = forest_leaks(det)

    t = time.perf_counter()
    ev = run_evidence(engine, cfg, region_id, "dev", series, det)
    secs["evidence"] = time.perf_counter() - t
    log.info("anomalies.dev.evidence", events=len(ev.evidence), seconds=round(secs["evidence"], 1))

    return DevOutputs(
        split=split,
        series=series,
        detection=det,
        injections=inj,
        cuts=cuts,
        forest_leaks=leaks,
        evidence=ev,
        band_issue_times=len(band.issue_times),
        band_valid_pairs=int(band.valid.sum()),
        checked_actuals=checked,
        seconds=secs,
    )


# --------------------------------------------------------------------------- storage


def interval_frame(det: Detection, split: Split) -> pd.DataFrame:
    """One row per interval: each calibrated detector's score, p, window size, tier and
    scored-period flag; each line's tier and flag."""
    masks = split.masks(det.run.ends)
    cols: dict[str, Any] = {"interval_end": det.run.ends}
    for d in det.run.recipe.detectors:
        cols[f"{d}__score"] = det.run.scores[d]
        cols[f"{d}__p"] = det.run.pvalues[d]
        cols[f"{d}__n_window"] = det.run.n_window[d]
        cols[f"{d}__tier"] = det.tiers[d]
        cols[f"{d}__in_period"] = masks[d]
    for d in LINES:
        cols[f"{d}__tier"] = det.tiers[d]
        cols[f"{d}__in_period"] = masks[d]
    return pd.DataFrame(cols)


def forest_frame(det: Detection) -> pd.DataFrame:
    rows = [
        {
            "month": f"{mm.month // 12}-{mm.month % 12 + 1:02d}",
            "month_start": mm.start,
            "train_first_end": mm.train_first_end,
            "train_last_end": mm.train_last_end,
            "n_train": mm.n_train,
        }
        for mm in sorted(det.run.state.models.values(), key=lambda m: m.month)
    ]
    return pd.DataFrame(rows)


def cut_frame(cuts: list[CutCheck]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cut_index": [c.index for c in cuts],
            "cut_end": [c.end for c in cuts],
            "n_differences": [len(c.differences) for c in cuts],
            "differences": ["; ".join(c.differences) for c in cuts],
            "seconds": [c.seconds for c in cuts],
        }
    )


def save_files(directory: Path, out: DevOutputs) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    interval_frame(out.detection, out.split).to_parquet(
        directory / "intervals.parquet", index=False, compression="zstd"
    )
    out.injections.to_parquet(directory / "injections.parquet", index=False)
    forest_frame(out.detection).to_parquet(directory / "forests.parquet", index=False)
    cut_frame(out.cuts).to_parquet(directory / "cuts.parquet", index=False)


def a4_counts(out: DevOutputs, cfg: AppConfig) -> dict[str, int]:
    phrases = cfg.anomalies.evidence.banned_phrases
    return {
        "events": len(out.evidence.evidence),
        "incomplete": sum(bool(missing_fields(ev)) for ev in out.evidence.evidence),
        "banned": sum(bool(banned_in(s, phrases)) for s in out.evidence.summaries),
    }


def run_config(out: DevOutputs, cfg: AppConfig, state: FreezeState, region_id: str) -> dict:
    a = cfg.anomalies
    det = out.detection
    events = det.events
    return {
        "split": out.split.name,
        "anomaly_config_hash": state.config_hash,
        "frozen_commit": state.frozen_commit,
        "region": region_id,
        "periods": {
            "market_first_day": str(out.split.market_first_day),
            "residual_first_day": str(out.split.residual_first_day),
            "end_day": str(out.split.end_day),
        },
        "series": {
            "intervals": len(out.series),
            "first_end": str(out.series.ends[0]),
            "last_end": str(out.series.ends[-1]),
        },
        "inputs": {
            "qr_batch": a.residual.dev_qr_batch,
            "mask_batch": a.explain.dev_batch,
            "band_issue_times": out.band_issue_times,
            "band_valid_pairs": out.band_valid_pairs,
            "checked_actuals": out.checked_actuals,
        },
        "counts": {
            "events": len(events),
            "lines_only": sum(e.lines_only for e in events),
            "severe": sum(e.tier == "severe" for e in events),
            "alerts": {d: int((t > 0).sum()) for d, t in det.tiers.items()},
            "forests": len(det.run.state.models),
            "injections": len(out.injections),
        },
        "a5_real": {
            "cuts": [
                {"index": c.index, "end": str(c.end), "differences": c.differences}
                for c in out.cuts
            ],
            "forest_leaks": out.forest_leaks,
        },
        "shap_gate": out.evidence.shap_gate.as_dict(),
        "a4": a4_counts(out, cfg),
        "seconds": {k: round(v, 1) for k, v in out.seconds.items()},
        "files": list(FILES),
    }


def insert_events(conn: Connection, run_id: int, region_id: str, out: DevOutputs) -> int:
    ends = out.series.ends
    rows = [
        {
            "run_id": run_id,
            "region": region_id,
            "split": out.split.name,
            "first": ends[e.first].to_pydatetime(),
            "last": ends[e.last].to_pydatetime(),
            "peak": ends[e.peak].to_pydatetime(),
            "tier": e.tier,
            "detectors": list(e.detectors),
            "min_p": e.min_p,
            "categories": list(e.categories),
            "evidence": json.dumps(ev, allow_nan=False),
            "summary": summary,
        }
        for e, ev, summary in zip(
            out.detection.events, out.evidence.evidence, out.evidence.summaries, strict=True
        )
    ]
    if rows:
        conn.execute(
            text(
                """
                INSERT INTO anomalies
                    (run_id, region_id, split, first_interval_end, last_interval_end,
                     peak_interval_end, tier, detectors, min_p, categories, evidence, summary)
                VALUES (:run_id, :region, :split, :first, :last, :peak, :tier, :detectors,
                        :min_p, :categories, CAST(:evidence AS jsonb), :summary)
                """
            ),
            rows,
        )
    return len(rows)


def store_dev(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    out: DevOutputs,
    state: FreezeState,
    batch_id: uuid.UUID | None = None,
) -> tuple[int, uuid.UUID, Path]:
    """Write the files, then the run row and its events in one transaction."""
    batch_id = batch_id or uuid.uuid4()
    directory = cfg.data_path(cfg.anomalies.output_dir) / str(batch_id)
    save_files(directory, out)
    config = run_config(out, cfg, state, region_id)
    eval_start = nem_midnight_utc(str(out.split.market_first_day))
    eval_end = nem_midnight_utc(str(out.split.end_day))
    try:
        with engine.begin() as conn:
            run_id = conn.execute(
                text(
                    """
                    INSERT INTO model_runs
                        (batch_id, model_name, model_kind, target, region_id, eval_start,
                         eval_end, holdout_start, config, git_sha, artifact_path, status,
                         finished_at)
                    VALUES (:batch, :name, 'anomaly', 'market', :region, :eval_start,
                            :eval_end, :holdout, CAST(:config AS jsonb), :sha, :artifact,
                            'success', now())
                    RETURNING run_id
                    """
                ),
                {
                    "batch": batch_id,
                    "name": MODEL_NAME,
                    "region": region_id,
                    "eval_start": eval_start.to_pydatetime(),
                    "eval_end": eval_end.to_pydatetime(),
                    "holdout": nem_midnight_utc(str(cfg.backtest.holdout_start)).to_pydatetime(),
                    "config": json.dumps(config, default=str, allow_nan=False),
                    "sha": state.sha,
                    "artifact": str(directory),
                },
            ).scalar_one()
            insert_events(conn, int(run_id), region_id, out)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return int(run_id), batch_id, directory


@dataclass(slots=True)
class DevRun:
    run_id: int
    batch_id: uuid.UUID
    directory: Path
    outputs: DevOutputs
    config: dict


def run_dev(engine: Engine, cfg: AppConfig, region_id: str) -> DevRun:
    state = check_preconditions(engine, cfg, region_id)
    log.info("anomalies.dev.start", hash=state.config_hash, sha=state.sha)
    out = compute_dev(engine, cfg, region_id)
    t = time.perf_counter()
    run_id, batch_id, directory = store_dev(engine, cfg, region_id, out, state)
    out.seconds["store"] = time.perf_counter() - t
    return DevRun(run_id, batch_id, directory, out, run_config(out, cfg, state, region_id))


def summary_lines(run: DevRun) -> list[str]:
    """Counts, checks and timings only; the criteria are computed by ``report-m4``."""
    c = run.config
    out = run.outputs
    cuts = c["a5_real"]["cuts"]
    gate = c["shap_gate"]
    return [
        f"stored anomaly run {run.run_id}, batch {run.batch_id} -> {run.directory}",
        f"   hash {c['anomaly_config_hash']}, frozen commit {c['frozen_commit']}",
        f"   series {c['series']['intervals']:,} intervals ({c['series']['first_end']} -> "
        f"{c['series']['last_end']}); band {c['inputs']['band_issue_times']:,} issue times, "
        f"actuals checked on {c['inputs']['checked_actuals']:,} pairs",
        f"   events {c['counts']['events']:,} (lines only {c['counts']['lines_only']:,}, "
        f"severe {c['counts']['severe']:,}); forests {c['counts']['forests']}; "
        f"injections {c['counts']['injections']:,}",
        "   real cuts: "
        + ", ".join(f"{x['end']} ({len(x['differences'])} differences)" for x in cuts)
        + f"; forest leaks {len(c['a5_real']['forest_leaks'])}",
        f"   E1 at demand peaks: {gate['n']} pairs, forecast mismatches "
        f"{gate['forecast_mismatches']}, sum failures {gate['sum_failures']}",
        f"   A4 counts: {c['a4']['incomplete']} incomplete, {c['a4']['banned']} with a banned "
        f"phrase, of {c['a4']['events']} events",
        "   seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in out.seconds.items()),
    ]


def cut_passed(run_config: dict) -> bool:
    real = run_config["a5_real"]
    return all(not c["differences"] for c in real["cuts"]) and not real["forest_leaks"]
